"""简单的神经网络为例,来实现求梯度的代码"""
import numpy as np
import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.funtions import cross_entropy_error,softmax,numerical_gradient

class simpleNet:
    def __init__(self) -> None:
        self.W = np.random.randn(2,3)
    
    def predict(self,x):
        return np.dot(x,self.W)

    def loss(self,x,t):
        z = self.predict(x)
        y =softmax(z)
        loss = cross_entropy_error(y,t)
        return loss 
    

    
if __name__ == "__main__":
    net = simpleNet()
    print(f"原始权重：{net.W}")
    x = np.array([0.6,0.9])
    p = net.predict(x)
    print(f"最大值索引：{np.argmax(p)}")
    print(f"预测结果：{p}")
    t = np.array([0,0,1])
    print(f"损失函数：{net.loss(x,t)}")

    f = lambda W: net.loss(x,t)

    # 求梯度
    dW = numerical_gradient(f,net.W)
    print(f"梯度：{dW}")

    