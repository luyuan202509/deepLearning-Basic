from turtle import back
import numpy as np
import sys, os

from unit4.train_neuralnet import grads, loss
# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    

from common.functions import *
from collections import OrderedDict

from common.layers import Affine,Relu,SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
    
       # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        
        self.lastLayer = SoftmaxWithLoss()
    
    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
    # x: 输入数据，t：监督数据
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1: t = np.argmax(t,axis=1)
        
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    def gradient(self,x,t):
        # forward
        self.loss(x,t)

        # backward 
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['b1'].dW
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['b2'].dW

        return grads






def print_param(net):
   
    for i in range(len(net.params)):
        #print(f"参数{net.params[i]}形状：{net.params[i].shape}")
       print(net.params['W1'].shape)
       print(net.params['b1'].shape)
       print(net.params['W2'].shape)
       print(net.params['b2'].shape)
def main():
    net = TwoLayerNet(input_size=784,hidden_size=100,output_size=10)
    print_param(net)
    x = np.random.rand(100,784)
    t = np.random.rand(100,10)
    grads = net.numerical_gradient(x,t)


if __name__ == '__main__':
    main()