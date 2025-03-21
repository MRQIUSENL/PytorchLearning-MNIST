# PytorchLearning

## 项目简介
本项目是一个用于学习和实践 PyTorch 的代码库，主要实现了经典的 LeNet 模型，并在 MNIST 数据集上进行训练和测试。通过本项目，你可以学习如何使用 PyTorch 构建和训练卷积神经网络（CNN）。

## 文件结构
以下是项目的文件结构及其作用：
PytorchLearning/
├── data/
│ └── MNIST/
│ └── raw/ # 存放 MNIST 数据集的原始文件
├── res/ # 存放训练结果或中间文件
├── model.py # LeNet 模型的定义
├── train.py # 训练模型的代码
├── testpho.png # 测试图片
├── README.md # 项目说明文档
└── requirements.txt # 项目依赖包列表

## 安装指南
1. 克隆本仓库：
   ```bash
   git clone https://github.com/ChillForrest/PytorchLearning.git
   pip install -r requirements.txt

## 参考文献
Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 1998.
