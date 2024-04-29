import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os
from tqdm import tqdm
from img2data_res import prepare_data_loaders
from sklearn.metrics import f1_score, accuracy_score


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"Epoch {epoch + 1} - {phase}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        if phase == 'train':
            scheduler.step()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, dataloader, device='cpu'):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    print(f'Test Accuracy: {acc:.4f}, Test F1 Score: {f1:.4f}')


def main():
    image_dir = "image"  # 指定图片文件夹的路径
    batch_size = 16  # 指定批量大小
    num_epochs = 50  # 指定训练的周期数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断使用GPU还是CPU

    # 调用 prepare_data_loaders 函数来获取数据加载器
    train_loader, val_loader, test_loader = prepare_data_loaders(image_dir, test_size=0.2, val_size=0.1, batch_size=batch_size)

    dataloaders = {'train': train_loader, 'val': val_loader}

    model = models.resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.layer4[2].conv3.requires_grad = True
    model.fc.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam([{'params': model.layer4[2].conv3.parameters(), 'lr': 0.001},
                      {'params': model.fc.parameters(), 'lr': 0.01}])
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    trained_model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs,
                                device=device)

    torch.save(trained_model.state_dict(), 'trained_resnet101.pth')

    # 测试模型
    test_model(trained_model, test_loader, device=device)


if __name__ == "__main__":
    main()


