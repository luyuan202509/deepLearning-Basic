import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from common.functions import sigmoid,softmax,cross_entropy_error
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    '''
        def forwald(self,x):
        return np.maximum(0,x)
    '''
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx 

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = sigmoid(x)
        self.out = out
        return out
    def backward(self,dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
        
    def forward(self,x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self,dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss():
    '''Softmax-with-Loss层的实现'''
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)/batch_size
        return dx