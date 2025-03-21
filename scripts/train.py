import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # 进度条显示
import model

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数配置
config = {
    "batch_size": 64,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "num_classes": 10
}

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 统一尺寸
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
])

# 加载数据集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["batch_size"],
    shuffle=False
)

# 初始化模型
model = model.LeNet(num_classes=config["num_classes"]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# 训练循环
best_acc = 0.0
for epoch in range(config["num_epochs"]):
    # 训练模式
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["num_epochs"]}')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计信息
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条信息
        pbar.set_postfix({
            'loss': running_loss / (total / config["batch_size"]),
            'acc': 100 * correct / total
        })

    # 验证模式
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_lenet.pth")

print("Training Complete!")
print(f"Best Test Accuracy: {best_acc:.2f}%")