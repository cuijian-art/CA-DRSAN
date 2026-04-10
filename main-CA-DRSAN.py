import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import numpy as np
import os
from torchvision.models import resnet50, ResNet50_Weights
import data_loader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ====================== 坐标注意力CA模块======================
class CA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        # 修复：正确的高度/宽度池化 + 维度拼接
        x_h = torch.mean(x, dim=3, keepdim=True)  # [B, C, H, 1]  高度池化
        x_w = torch.mean(x, dim=2, keepdim=True)  # [B, C, 1, W]  宽度池化

        # 修复：在dim=2拼接，得到 [B, C, H+W, 1]
        x_cat = torch.cat([x_h, x_w.permute(0, 1, 3, 2)], dim=2)
        x_cat = self.conv1(x_cat)
        x_cat = self.bn1(x_cat)
        x_cat = self.act(x_cat)

        # 修复：正确拆分H和W
        x_h, x_w = torch.split(x_cat, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        channel_att = (a_h * a_w).mean(dim=[2, 3])
        out = x * a_h.expand_as(x) * a_w.expand_as(x)
        return out, channel_att


# 带CA的ResNet Bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_ca=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.use_ca = use_ca
        if self.use_ca:
            self.ca = CA(planes * self.expansion)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        channel_att = None
        if self.use_ca:
            out, channel_att = self.ca(out)

        out += residual
        out = self.relu(out)
        return out, channel_att


# 带CA的ResNet50特征提取器
class CAResNet50(nn.Module):
    def __init__(self, pretrained=True, use_ca=True):
        super().__init__()
        self.inplanes = 64
        self.use_ca = use_ca

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if pretrained:
            pretrained_dict = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'ca' not in k}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_ca=self.use_ca))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_ca=self.use_ca))
        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        final_channel_att = None
        for block in self.layer1:
            x, att = block(x)
            if att is not None:
                final_channel_att = att
        for block in self.layer2:
            x, att = block(x)
            if att is not None:
                final_channel_att = att
        for block in self.layer3:
            x, att = block(x)
            if att is not None:
                final_channel_att = att
        for block in self.layer4:
            x, att = block(x)
            if att is not None:
                final_channel_att = att

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x, final_channel_att


# 多核高斯核函数
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val) / len(kernel_val)


# 多层MK-LMMD损失计算
def mk_lmmd(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None, num_classes=31):
    n_s = source.size(0)
    n_t = target.size(0)
    if n_s == 0 or n_t == 0:
        return torch.tensor(0.0, device=source.device)

    kernel = guassian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    ss = kernel[:n_s, :n_s]
    tt = kernel[n_s:, n_s:]
    st = kernel[:n_s, n_s:]

    loss = 0.0
    for c in range(num_classes):
        s_mask = (s_label == c).float()
        t_mask = (t_label == c).float()
        n_s_c = s_mask.sum()
        n_t_c = t_mask.sum()
        if n_s_c == 0 or n_t_c == 0:
            continue

        w_ss = torch.outer(s_mask, s_mask) / (n_s_c ** 2)
        w_tt = torch.outer(t_mask, t_mask) / (n_t_c ** 2)
        w_st = torch.outer(s_mask, t_mask) / (n_s_c * n_t_c)

        loss += (w_ss * ss).sum() + (w_tt * tt).sum() - 2 * (w_st * st).sum()

    return loss / num_classes


# CA-DRSAN主模型
class CADRSAN(nn.Module):
    def __init__(self, num_classes=31, pretrained=True, use_ca=True):
        super().__init__()
        self.feature_layers = CAResNet50(pretrained=pretrained, use_ca=use_ca)
        feature_dim = 2048

        self.fc1 = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.cls_fc = nn.Linear(256, num_classes)

    def forward(self, source, target=None):
        s_feat, s_att = self.feature_layers(source)
        s_fc1 = self.fc1(s_feat)
        s_fc2 = self.fc2(s_fc1)
        s_fc3 = self.fc3(s_fc2)
        s_pred = self.cls_fc(s_fc3)

        if target is None:
            return s_pred

        t_feat, t_att = self.feature_layers(target)
        t_fc1 = self.fc1(t_feat)
        t_fc2 = self.fc2(t_fc1)
        t_fc3 = self.fc3(t_fc2)
        t_pred = self.cls_fc(t_fc3)

        return s_pred, t_pred, [s_fc1, s_fc2, s_fc3], [t_fc1, t_fc2, t_fc3], s_att, t_att

    def predict(self, x):
        feat, _ = self.feature_layers(x)
        fc1 = self.fc1(feat)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        return self.cls_fc(fc3)


def load_data(root_path, src, tar, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    loader_tar = data_loader.load_training(root_path, tar, batch_size, kwargs)
    loader_tar_test = data_loader.load_testing(
        root_path, tar, batch_size, kwargs)
    return loader_src, loader_tar, loader_tar_test


# 计算源域类中心和类平均注意力
def compute_source_centers(model, source_loader, num_classes=31, device='cuda'):
    model.eval()
    class_centers = torch.zeros(num_classes, 256, device=device)
    class_attentions = torch.zeros(num_classes, 2048, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for data, label in source_loader:
            data = data.to(device)
            label = label.to(device)
            feat, att = model.feature_layers(data)
            fc3 = model.fc3(model.fc2(model.fc1(feat)))
            for i in range(label.size(0)):
                c = label[i]
                class_centers[c] += fc3[i]
                class_attentions[c] += att[i]
                class_counts[c] += 1

    class_counts[class_counts == 0] = 1
    class_centers /= class_counts.unsqueeze(1)
    class_attentions /= class_counts.unsqueeze(1)
    return class_centers, class_attentions


# 动态阈值计算
def get_dynamic_thresholds(epoch, total_epoch, tau_p_min=0.5, tau_p_max=0.9, tau_d_min=2.0, tau_d_max=10.0):
    tau_p = tau_p_min + (tau_p_max - tau_p_min) * (epoch / total_epoch)
    tau_d = tau_d_max - (tau_d_max - tau_d_min) * (epoch / total_epoch)
    return tau_p, tau_d


def train_epoch(epoch, model, dataloaders, optimizer, args, class_centers, class_attentions, tau_p, tau_d):
    model.train()
    source_loader, target_train_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)

    total_cls = 0.0
    total_lmmd = 0.0
    total_att = 0.0

    for i in range(1, num_iter):
        data_source, label_source = next(iter_source)
        data_target, _ = next(iter_target)
        if i % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)

        data_source = data_source.cuda()
        label_source = label_source.cuda()
        data_target = data_target.cuda()

        optimizer.zero_grad()
        s_pred, t_pred, s_feats, t_feats, s_att, t_att = model(data_source, data_target)

        # 源域分类损失
        loss_cls = F.cross_entropy(s_pred, label_source)

        # 高可靠伪标签筛选
        t_conf = F.softmax(t_pred, dim=1).max(dim=1)[0]
        t_pseudo = t_pred.argmax(dim=1)
        t_fc3 = t_feats[-1]
        t_dist = torch.norm(t_fc3 - class_centers[t_pseudo], p=2, dim=1)
        reliable_mask = (t_conf >= tau_p) & (t_dist <= tau_d)
        reliable_num = reliable_mask.sum().item()

        loss_lmmd = 0.0
        loss_att = 0.0

        if reliable_num > 0:
            # 计算三层MK-LMMD损失
            reliable_t_labels = t_pseudo[reliable_mask]
            reliable_t_feats = [f[reliable_mask] for f in t_feats]
            for l in range(3):
                loss_lmmd += mk_lmmd(s_feats[l], reliable_t_feats[l], label_source, reliable_t_labels,
                                     num_classes=args.nclass)
            loss_lmmd /= 3

            # 计算注意力一致性损失
            reliable_t_att = t_att[reliable_mask]
            for j in range(reliable_num):
                c = reliable_t_labels[j]
                loss_att += F.l1_loss(reliable_t_att[j], class_attentions[c])
            loss_att /= reliable_num

        # 总损失
        lambd = 2 / (1 + math.exp(-10 * epoch / args.nepoch)) - 1
        loss = loss_cls + lambd * (args.weight * loss_lmmd + args.alpha * loss_att)

        loss.backward()
        optimizer.step()

        total_cls += loss_cls.item()
        total_lmmd += loss_lmmd.item() if isinstance(loss_lmmd, torch.Tensor) else loss_lmmd
        total_att += loss_att.item() if isinstance(loss_att, torch.Tensor) else loss_att

        if i % args.log_interval == 0:
            avg_cls = total_cls / args.log_interval
            avg_lmmd = total_lmmd / args.log_interval
            avg_att = total_att / args.log_interval
            avg_total = avg_cls + lambd * (args.weight * avg_lmmd + args.alpha * avg_att)
            print(
                f'Epoch [{epoch:2d}] Iter [{i:3d}] Total: {avg_total:.4f} Cls: {avg_cls:.4f} MK-LMMD: {avg_lmmd:.4f} Att: {avg_att:.4f} Reliable: {reliable_num}/{data_target.size(0)} tau_p: {tau_p:.2f} tau_d: {tau_d:.2f}')
            total_cls = total_lmmd = total_att = 0.0


def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            pred = model.predict(data)
            test_loss += F.cross_entropy(pred, target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(dataloader)
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')
    return correct


def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default=r'F:\迁移学习代码\QUGS')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='EFA')
    parser.add_argument('--tar', type=str,
                        help='Target domain', default='TEST')
    parser.add_argument('--nclass', type=int,
                        help='Number of classes', default=31)
    parser.add_argument('--batch_size', type=int,
                        help='batch size', default=32)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=200)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.001, 0.01, 0.01])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=30)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2021)
    parser.add_argument('--weight', type=float,
                        help='Weight for MK-LMMD loss', default=0.5)
    parser.add_argument('--alpha', type=float,
                        help='Weight for attention consistency loss', default=0.1)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-4)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=10)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    parser.add_argument('--tau_p_min', type=float, default=0.5)
    parser.add_argument('--tau_p_max', type=float, default=0.9)
    parser.add_argument('--tau_d_min', type=float, default=2.0)
    parser.add_argument('--tau_d_max', type=float, default=10.0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataloaders = load_data(args.root_path, args.src, args.tar, args.batch_size)
    model = CADRSAN(num_classes=args.nclass).cuda()

    correct = 0
    stop = 0

    optimizer = torch.optim.SGD([
        {'params': model.feature_layers.parameters()},
        {'params': model.fc1.parameters(), 'lr': args.lr[1]},
        {'params': model.fc2.parameters(), 'lr': args.lr[1]},
        {'params': model.fc3.parameters(), 'lr': args.lr[1]},
        {'params': model.cls_fc.parameters(), 'lr': args.lr[2]},
    ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)

    for epoch in range(1, args.nepoch + 1):
        stop += 1
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = args.lr[min(index, len(args.lr) - 1)] / math.pow((1 + 10 * (epoch - 1) / args.nepoch),
                                                                                 0.75)

        print(f'\n===== Epoch {epoch}/{args.nepoch} =====')
        class_centers, class_attentions = compute_source_centers(model, dataloaders[0], num_classes=args.nclass)
        tau_p, tau_d = get_dynamic_thresholds(epoch, args.nepoch, args.tau_p_min, args.tau_p_max, args.tau_d_min,
                                              args.tau_d_max)

        train_epoch(epoch, model, dataloaders, optimizer, args, class_centers, class_attentions, tau_p, tau_d)
        t_correct = test(model, dataloaders[-1])

        if t_correct > correct:
            correct = t_correct
            stop = 0
            torch.save(model, 'ca_drsan.pth')
        print(f'{args.src}->{args.tar} Best Acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')

        if stop >= args.early_stop:
            print(f'Final Test Acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
            break