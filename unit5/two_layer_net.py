import os,sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from common.layers import Affine,Relu,SoftmaxWithLoss
from collections import OrderedDict
from common.functions import numerical_gradient
class TwoLayerNet:
    '''2层神经网络'''
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        '''初始化参数'''
        self.params = {}
        self.params = {
            'W1':weight_init_std * np.random.randn(input_size,hidden_size),
            'b1':np.zeros(hidden_size),
            'W2':weight_init_std * np.random.randn(hidden_size,output_size),
            'b2':np.zeros(output_size)
        }
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
    
    def predict(self,x):
        '''预测'''
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self,x,t):
        '''损失函数'''
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self,x,t):
        '''准确率 '''
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self,x,t):
        '''数值梯度下降'''
        loss = lambda W: self.loss(x,t)
        print('梯度计算中...')
        grads = {}
        grads['W1'] = numerical_gradient(loss,self.params['W1'])
        grads['b1'] = numerical_gradient(loss,self.params['b1'])
        grads['W2'] = numerical_gradient(loss,self.params['W2'])
        grads['b2'] = numerical_gradient(loss,self.params['b2'])
        return grads
    
    def gradient(self,x,t):
        ''' 数值梯度下降-高速版'''
        # forworld
        self.loss(x,t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            dout = layer.backward(dout)
        
        # 填写结果
        grads = {} 
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db 
        grads['W2'] = self.layers['Affine2'].dW 
        grads['b2'] = self.layers['Affine2'].db
        
        return grads
        