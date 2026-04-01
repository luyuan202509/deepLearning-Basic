import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,random_split

from torchvision.datasets import MNIST
from torchvision import transforms

from simple_dense_net import SimpleDenseNet

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')




if __name__ == "__main__":
    model = SimpleDenseNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数,适用于多分类问题
    
    dataset = MNIST(root='/Users/luyuan/neulife/pyproject/deepLearning-Basic/data', 
                    train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


    for epoch in range(10):
        for data in train_loader:
            inputs,labels = data
            optimizer.zero_grad() # 清零梯度
            output = model(inputs.to(device)) # 前向传播
            loss = criterion(output, labels.to(device)) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            #if batch_idx % 100 == 0: # 每100个批次打印一次损失
            #    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
        print(f'Epoch [{epoch +1} / 10],loss: {loss.item():.4f}')