# coding: utf-8
import sys, os

#sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
from common.functions import softmax, cross_entropy_error,numerical_gradient



class simpleNet:
    '''单层的神经网络'''
    def __init__(self):
        self.W = np.random.randn(2,3)
    
    def predict(self,x):
        return np.dot(x,self.W)
    
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        return loss

        
        

# 预测学习
net = simpleNet()
print(f"权重：{net.W}")
x = np.array([0.6, 0.9])
p =  net.predict(x)

t = np.array([0, 0, 1]) # 正确解标签 
loss  = net.loss(x,t)

#函数
'''
def f(W):
    loss = net.loss(x,t)
    return loss
'''
f = lambda W: net.loss(x,t)


dw =  numerical_gradient(f,net.W)
print(f'导数(梯度)：：{dw}')

