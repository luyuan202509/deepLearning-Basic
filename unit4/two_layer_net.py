import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from common.functions import cross_entropy_error, sigmoid, sigmoid_grad, softmax,numerical_gradient
class TwoLayerNet:
    '''2层神经网络'''
    
   # network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        '''初始化权重'''
        self.params = {}
        self.params = {
            'W1':weight_init_std * np.random.randn(input_size,hidden_size),
            'b1':np.zeros(hidden_size),
            'W2':weight_init_std * np.random.randn(hidden_size,output_size),
            'b2':np.zeros(output_size)
        }
    
    def predict(self,x):
        '''预测'''
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)
        return y
    
    # x:输入数据，t:监督数据(正确数据)
    def loss(self,x,t):
        '''损失函数'''
        y = self.predict(x)
        loss = cross_entropy_error(y,t)
        return loss
    
    ''' 
    def accuracy(self,x,t):
        """准确率"""
        accuracy_cnt = 0;
        if x.ndim == 1:
            y = self.predict(x)
            y = np.argmax(y,axis=1)
            t = np.argmax(t,axis=1)
            accuracy_cnt = np.sum(y == t) / float(x.shape[0])
            return accuracy_cnt
    '''
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    # 计算权重参数的梯度 x:输入数据，t:监督数据(正确数据)            
    def numerical_gradient(self,x,t):
        loss_W  = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
        return grads

    #计算权重参数的梯度-numerical_gradient()的高速版
    def gradient(self,x,t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

