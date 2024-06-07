import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch.cuda.amp import GradScaler, autocast  # 用于混合精度训练

from img2data import prepare_data_loaders
from model_vgg_19 import vgg19_Net

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
    use_cuda = torch.cuda.is_available()
    scaler = GradScaler() if use_cuda else None  # 仅在 CUDA 可用时初始化 GradScaler

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        start_time = time.time()  # 开始计时

        # 使用 tqdm 包装迭代器来显示进度
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True)

        for inputs, labels in progress:
            # 将数据移动到当前设备上（CPU或GPU）
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            if use_cuda:
                with autocast():  # 使用混合精度训练
                    # 前向传播
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # 反向传播
                loss.backward()
                optimizer.step()

            # 更新总损失
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            # 更新进度条描述信息
            progress.set_description(f'Epoch {epoch + 1}/{num_epochs} Loss: {running_loss / len(train_loader.dataset):.4f}')

        epoch_duration = time.time() - start_time  # 计算一个epoch的耗时
        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_predictions / total_predictions * 100
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Duration: {epoch_duration:.2f}s')

        # 验证模型
        val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, criterion)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # 更新学习率
        scheduler.step(val_loss)

    print('Training complete')

def evaluate_model(model, data_loader, criterion):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    total_correct = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # 在评估阶段不计算梯度
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / len(data_loader.dataset) * 100
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    return avg_loss, accuracy, f1

print("action")

# 模型定义
model = vgg19_Net(in_img_rgb=3, in_img_size=224, out_class=2, in_fc_size=512*7*7)  # 根据实际情况调整 in_fc_size
print("model finish prepare")
# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
# 检测是否有可用的GPU，如果没有，则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型发送到设备

print("start prepare images")

# 数据预处理
image_directory = 'image'  # 图像文件夹路径
batch_size = 8  # 设置批处理大小
train_loader, val_loader, test_loader = prepare_data_loaders(image_directory, batch_size=batch_size)

print("finish prepare images")

# 训练模型
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

# 评估模型
test_loss, test_accuracy, test_f1 = evaluate_model(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test F1 Score: {test_f1:.4f}')




