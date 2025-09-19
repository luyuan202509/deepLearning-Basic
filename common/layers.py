
import numpy as np 
from common.functions import softmax ,cross_entropy_error


class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b
        return out
    
    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        dy = np.dot(self.x.T,dout)
        db = np.sum(dout,axis=0)
        return dx,dy,db

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 损失
        self.y = None   #  softmax的输出
        self.t = None  #  监督数据（one-hot vector）
    
    def forward(self,x,t):
        '''前向传播'''
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
    
    def backward(self,dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
