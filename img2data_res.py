import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def prepare_data_loaders(image_dir, test_size=0.2, val_size=0.1, batch_size=4):
    # 获取文件夹中的类别
    classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]

    # 读取图像文件名和提取标签
    images = []
    labels = []
    for class_idx, class_name in tqdm(enumerate(classes), desc="Reading images and labels"):
        class_images = [os.path.join(class_name, f) for f in os.listdir(os.path.join(image_dir, class_name)) if
                        os.path.isfile(os.path.join(image_dir, class_name, f))]
        images += class_images
        labels += [class_idx] * len(class_images)

    # 划分测试集和临时训练集（训练集 + 验证集）
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=42
    )

    # 从临时训练集中进一步划分出训练集和验证集
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images, train_val_labels, test_size=val_size, stratify=train_val_labels, random_state=42
    )

    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 数据集类定义
    class CustomImageDataset(Dataset):
        def __init__(self, image_filenames, labels, img_dir, transform=None):
            self.image_filenames = image_filenames
            self.labels = labels
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.image_filenames)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.image_filenames[idx])
            try:
                image = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                print(f"File not found: {img_path}")
                return None, None
            if self.transform:
                image = self.transform(image)

            label = self.labels[idx]
            return image, label

    # 创建训练、验证和测试数据集
    train_dataset = CustomImageDataset(train_images, train_labels, image_dir, transform)
    val_dataset = CustomImageDataset(val_images, val_labels, image_dir, transform)
    test_dataset = CustomImageDataset(test_images, test_labels, image_dir, transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
