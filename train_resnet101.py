import torch
import torch.nn as nn
from torchvision import models
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from img2data import prepare_data_loaders
from torchvision.models import resnet101, ResNet101_Weights
from torch.cuda.amp import GradScaler, autocast

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, patience=5, device='cpu'):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    no_improve_epochs = 0  # Early stopping tracker
    scaler = GradScaler() if device.type == 'cuda' else None  # 仅在 CUDA 可用时初始化 GradScaler

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
            total_samples = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"Epoch {epoch + 1} - {phase}"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                if device.type == 'cuda' and phase == 'train':
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples * 100

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping: No improvement in validation accuracy")
            break

        scheduler.step()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}')
    return test_acc, test_f1


def main():
    image_dir = "image"
    batch_size = 16
    num_epochs = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保这里正确地接收了三个数据加载器
    train_loader, val_loader, test_loader = prepare_data_loaders(image_dir, test_size=0.2, val_size=0.1, batch_size=batch_size)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # 更新模型加载方式，避免使用过时的参数
    model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),  # 加入 Dropout
        nn.Linear(num_ftrs, 2)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # 使用 L2 正则化
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)  # 调度学习率

    trained_model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs, patience=5, device=device)

    test_model(trained_model, test_loader, device=device)

    torch.save(trained_model.state_dict(), 'trained_resnet101_optimized.pth')

if __name__ == "__main__":
    main()


