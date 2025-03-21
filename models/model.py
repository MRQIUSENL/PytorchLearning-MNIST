import torch
import torch.nn as nn
import torch.nn.functional as F
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet,self).__init__()
        #卷积层和池化层
        self.conv1=nn.Conv2d(1,6,kernel_size=5)
        self.pool=nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(6,16,kernel_size=5)
        #全连接层(假设输入图像大小为28x28)
        self.fc1=nn.Linear(16*4*4,120)#输入尺寸根据卷积结果计算
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,num_classes)#输出层


    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x))) #[28x28]-->[24*24]-->[12x12]
        x=self.pool(F.relu(self.conv2(x))) #[12x12]-->[8*8]-->[4*4]
        #展平特征向量
        x=x.view(-1,16*4*4)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x) #输出logits（未归一化）
        return x
    
# net=LeNet()
# print(net)
# test_input=torch.randn(1,1,28,28)
# test_output=net(test_input)
# print(test_output.shape)
