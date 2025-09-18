# 正确计算项目根目录
import os,sys 
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
review_dir = os.path.dirname(current_dir)  # review目录
project_root = os.path.dirname(review_dir)  # 项目根目录
sys.path.append(project_root)
import numpy as np
from common.functions import cross_entropy_error,softmax

class simpleNet:
    def __init__(self) -> None:
        self.W = np.random.randn(2,3) # 用高斯分布进行初始化
    
    def predict(self,x):
        return np.dot(x,self.W)
    
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        return loss


if __name__ == "__main__":
    net = simpleNet()
    print(f'网络初始值：{net.W}')
    x = np.array([0.6,0.9])
    p = net.predict(x)
    print(f'预测结果：{p}')
    t = np.array([0,0,1])
    print(f'损失函数：{net.loss(x,t)}')
