import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import numpy as np
import os
import data_loader



def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss



class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = F.relu(self.branch1(x), inplace=True)
        branch2 = F.relu(self.branch2(x), inplace=True)
        branch3 = F.relu(self.branch3(x), inplace=True)
        branch4 = F.relu(self.branch4(x), inplace=True)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogLeNetFeature(nn.Module):
    def __init__(self):
        super(GoogLeNetFeature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x


class DSAN(nn.Module):
    def __init__(self, num_classes=31, bottleneck=True):
        super(DSAN, self).__init__()
        self.feature_layers = GoogLeNetFeature()
        self.bottleneck = bottleneck

        if self.bottleneck:
            self.bottle = nn.Sequential(
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True)
            )
            self.cls_fc = nn.Linear(256, num_classes)
        else:
            self.cls_fc = nn.Linear(1024, num_classes)

    def forward(self, source, target, label_source):
        s_feature = self.feature_layers(source)
        if self.bottleneck:
            s_feature = self.bottle(s_feature)
        s_pred = self.cls_fc(s_feature)

        t_feature = self.feature_layers(target)
        if self.bottleneck:
            t_feature = self.bottle(t_feature)

        loss_lmmd = mmd_linear(s_feature, t_feature)
        return s_pred, loss_lmmd

    def predict(self, x):
        x = self.feature_layers(x)
        if self.bottleneck:
            x = self.bottle(x)
        x = self.cls_fc(x)
        return x


def load_data(root_path, src, tar, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    loader_tar = data_loader.load_training(root_path, tar, batch_size, kwargs)
    loader_tar_test = data_loader.load_testing(
        root_path, tar, batch_size, kwargs)
    return loader_src, loader_tar, loader_tar_test


def train_epoch(epoch, model, dataloaders, optimizer):
    model.train()
    source_loader, target_train_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)
    for i in range(1, num_iter):
        data_source, label_source = next(iter_source)
        data_target, _ = next(iter_target)
        if i % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target = data_target.cuda()

        optimizer.zero_grad()
        label_source_pred, loss_lmmd = model(
            data_source, data_target, label_source)
        loss_cls = F.nll_loss(F.log_softmax(
            label_source_pred, dim=1), label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1
        loss = loss_cls + args.weight * lambd * loss_lmmd

        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print(
                f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}')


def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            pred = model.predict(data)
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
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
    parser.add_argument('--root_path', type=str, help='Root path for dataset', default=r'F:\迁移学习代码\QUGS')
    parser.add_argument('--src', type=str, help='Source domain', default='EFM')
    parser.add_argument('--tar', type=str, help='Target domain', default='TEST')
    parser.add_argument('--nclass', type=int, help='Number of classes', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--nepoch', type=int, help='Total epoch num', default=200)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.001, 0.01, 0.01])
    parser.add_argument('--early_stop', type=int, help='Early stoping number', default=30)
    parser.add_argument('--seed', type=int, help='Seed', default=2021)
    parser.add_argument('--weight', type=float, help='Weight for adaptation loss', default=0.5)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float, help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--log_interval', type=int, help='Log interval', default=10)
    parser.add_argument('--gpu', type=str, help='GPU ID', default='0')
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
    model = DSAN(num_classes=args.nclass).cuda()

    correct = 0
    stop = 0

    if args.bottleneck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': args.lr[1]},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[2]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[1]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)

    for epoch in range(1, args.nepoch + 1):
        stop += 1
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = args.lr[index] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)

        train_epoch(epoch, model, dataloaders, optimizer)
        t_correct = test(model, dataloaders[-1])
        if t_correct > correct:
            correct = t_correct
            stop = 0
            torch.save(model, 'model_googlenet.pkl')
        print(
            f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')

        if stop >= args.early_stop:
            print(f'Final test acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
            break