import numpy as np
from functions import sigmoid
class Relu:
    def __init__(self):
        self.mask = None

    def froward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    '''
        def forwald(self,x):
        return np.maximum(0,x)
    '''
    def backword(self,dout):
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
    def backword(self,dout):
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

    def backword(self,dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx