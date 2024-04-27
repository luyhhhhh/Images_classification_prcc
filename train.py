import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm

from img2data import prepare_data_loaders
from test import vgg19_Net


def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()  # 开始计时

        # 使用 tqdm 包装迭代器来显示进度
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_eluypochs}', leave=True)
        for inputs, labels in progress:
            # 将数据移动到当前设备上（CPU或GPU）
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 更新总损失
            running_loss += loss.item() * inputs.size(0)

            # 更新进度条描述信息
            progress.set_description(f'Epoch {epoch + 1}/{num_epochs} Loss: {running_loss / len(train_loader.dataset):.4f}')

        epoch_duration = time.time() - start_time  # 计算一个epoch的耗时
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Duration: {epoch_duration:.2f}s')

    print('Training complete')


def evaluate_model(model, test_loader, criterion):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    total_correct = 0

    with torch.no_grad():  # 在评估阶段不计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = total_correct / len(test_loader.dataset) * 100
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


print("action")

# 模型定义
model = vgg19_Net(in_img_rgb=3, in_img_size=224, out_class=2, in_fc_size=512*7*7)  # 根据实际情况调整 in_fc_size
print("model finish prepare")
# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 检测是否有可用的GPU，如果没有，则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型发送到设备

print("start prepare images")

# 数据预处理
image_directory = 'image'  # 图像文件夹路径
train_loader, test_loader = prepare_data_loaders(image_directory)

print("finish prepare images")

# 训练模型
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# 评估模型
evaluate_model(model, test_loader, criterion)