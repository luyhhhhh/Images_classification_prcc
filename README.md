# Images_classification_prcc
### 项目代码使用教程

#### 环境准备

在运行本项目代码之前，需要确保已经安装以下环境和依赖包：

1. **操作系统**：Windows、macOS或Linux
2. **Python版本**：Python 3.8或更高版本
3. **深度学习框架**：PyTorch

首先，建议使用 `conda` 创建一个虚拟环境来管理项目的依赖包。

##### 使用 `conda` 创建虚拟环境

```
# 创建虚拟环境
conda create -n myname python=3.8

# 激活虚拟环境
conda activate myname
```

##### 安装依赖包

运行以下命令安装依赖包：

```
pip install -r requirements.txt
```

#### 数据集准备

确保数据集已经下载并放置在合适的目录中。例如，假设数据集位于 `image/` 目录下，目录结构如下：

```
image/
├── 1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── 2/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

#### 代码结构

项目代码包含以下几个主要文件：

1. `train_vgg_19.py`和`train_resnet_101.py`：用于训练vgg_19或resnet_101模型
2. `model_vgg_19.py`：定义了VGG19模型结构
3. `img2data.py`：用于数据预处理和加载

#### 运行代码

##### 训练模型

首先，确保数据集路径和其他参数在 `train.py` 文件中已正确设置。然后，运行以下命令开始训练模型：

```
python train_vgg_19.py    //调用vgg19模型
python train_resnet_101.py    //调用resnet101模型
```

在训练过程中，每个训练周期结束后会输出训练损失和准确率，以及验证损失和准确率。训练完成后，会在测试集上评估模型，并输出测试准确率和F1-score。

##### 评估模型

​	训练完成后，模型会自动在测试集上进行评估，并输出测试准确率和F1-score。如果需要单独运行评估，可以调用 `evaluate_model` 函数，并传入相应的模型和数据加载器。

### 环境安装文件

运行以下命令来创建并配置环境：

```
conda env create -f environment.yml
conda activate myname
```

通过以上步骤，您可以成功创建并配置运行本项目代码所需的环境。
