import numpy as np
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