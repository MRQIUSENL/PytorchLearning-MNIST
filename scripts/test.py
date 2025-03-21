import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
import model

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像预处理（需与训练时一致）
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整尺寸
    transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
])

# 加载训练好的模型
model = model.LeNet(num_classes=10).to(device)
model.load_state_dict(torch.load('best_lenet.pth', map_location=device))
model.eval()  # 设置为评估模式


def predict_image(image_path):
    # 加载并预处理图像
    image = Image.open(image_path)
    plt.imshow(image.convert('L'), cmap='gray')  # 显示原图

    # 应用预处理
    input_tensor = transform(image).unsqueeze(0).to(device)  # 添加batch维度

    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        pred_class = torch.argmax(probabilities).item()

    # 显示结果
    plt.title(f'Predicted: {pred_class}\nConfidence: {probabilities[pred_class]:.2%}')
    plt.axis('off')
    plt.show()

    # 打印详细概率
    print("\nClassification results:")
    for i in range(10):
        print(f"Number {i}: {probabilities[i]:.2%}")

    return pred_class


# 使用示例（替换为你的图片路径）
test_image_path = "testpho.png"  # 示例图片
prediction = predict_image(test_image_path)