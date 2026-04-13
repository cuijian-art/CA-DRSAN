import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import shufflenet_v2_x1_0
from my_dataset import MyDataSet
from utils import train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    # 直接加载训练集和验证集路径
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []

    # 假设你已经手动划分了数据集，直接使用文件夹路径加载
    # 假设训练集和验证集路径分别是 F:/data_set/SHM-GADFY/train 和 F:/data_set/SHM-GADFY/val
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")

    # 遍历训练集和验证集文件夹，手动加载图像路径和标签
    for label, class_name in enumerate(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.endswith(('.jpg', '.png')):  # 假设只加载 .jpg 和 .png 文件
                    train_images_path.append(os.path.join(class_path, file))
                    train_images_label.append(label)

    for label, class_name in enumerate(os.listdir(val_dir)):
        class_path = os.path.join(val_dir, class_name)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.endswith(('.jpg', '.png')):  # 假设只加载 .jpg 和 .png 文件
                    val_images_path.append(os.path.join(class_path, file))
                    val_images_label.append(label)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.223, 0.709, 0.677], [0.150, 0.088, 0.167])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.223, 0.709, 0.677], [0.150, 0.088, 0.167])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print(f'Using {nw} dataloader workers every process')

    # 加载数据
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 初始化模型
    model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)

    # 加载预训练权重
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError(f"not found weights file: {args.weights}")

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "fc" not in name:
                para.requires_grad_(False)

    # 定义优化器和学习率调度器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train: 计算训练损失和训练精度
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算训练损失和精度
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = correct_train / total_train
        train_loss /= len(train_loader)

        # validate: 计算验证损失和验证精度
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = model(val_images)
                loss = torch.nn.CrossEntropyLoss()(outputs, val_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == val_labels).sum().item()
                total_val += val_labels.size(0)

        val_accuracy = correct_val / total_val
        val_loss /= len(val_loader)

        # 打印训练和验证结果
        print(f"[epoch {epoch + 1}] "
              f"train_loss: {train_loss:.4f} train_accuracy: {train_accuracy:.4f} "
              f"val_loss: {val_loss:.4f} val_accuracy: {val_accuracy:.4f}")

        # TensorBoard 记录
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_accuracy", train_accuracy, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_accuracy", val_accuracy, epoch)

        # 保存模型
        torch.save(model.state_dict(), f"./weights/shufflenetv2-.pth")

        # 更新学习率
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="F:/data_set/BA")

    # 预训练权重路径
    parser.add_argument('--weights', type=str, default='./shufflenetv2_x1.pth', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)