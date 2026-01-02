import numpy as np
import sys, os
# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from common.functions import softmax, cross_entropy_error,numerical_gradient
#from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 随机初始化

    def predict(self,x):
        return np.dot(x, self.W)
    
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        return loss


def main():
    net = simpleNet()
    print("权重参数：",net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print("预测结果：",p)
    print("预测结果最大索引：",np.argmax(p))

def main2():
    net = simpleNet()
    print("权重参数：",net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print("预测结果：",p)
    print("预测结果最大索引：",np.argmax(p))

    t = np.array([0, 0, 1]) # 正确解标签
    loss = net.loss(x, t)
    print("损失函数值：",loss)

    def f(W):
        return net.loss(x, t)

    
    dW = numerical_gradient(f, net.W)
    print("该神经网络损失函数的梯度",dW)

if __name__ == "__main__":
    #main()
    main2()