import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
from tqdm import tqdm
from img2data import prepare_data_loaders


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cpu'):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # 每个epoch有两个训练阶段：训练和验证
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            # 使用 tqdm 封装 dataloader
            for inputs, labels in tqdm(dataloaders[phase], desc=f"Epoch {epoch+1} - {phase}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清空梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 如果是训练阶段，进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 深拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print('Best val Acc: {:4f}'.format(best_acc))

    # 载入最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


def main():
    image_dir = "image"  # 您的图像数据目录
    batch_size = 16
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_loader, test_loader = prepare_data_loaders(image_dir, test_size=0.2, batch_size=batch_size)
    dataloaders = {'train': train_loader, 'val': test_loader}

    # 加载预训练的 ResNet101 模型并修改全连接层
    model = models.resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # 初始时冻结所有参数

    # 修改全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 假设分类任务为2类

    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.fc.parameters(), lr=0.001)  # 仅优化全连接层

    # 训练和评估模型
    trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, device=device)

    # 保存模型
    torch.save(trained_model.state_dict(), 'trained_resnet101.pth')

if __name__ == "__main__":
    main()
