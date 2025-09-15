import sys,os
sys.append(os.path.dirname(os.path.abspath(__file__)) + "..")
import numpy as np

class SGD:
    def __init__(self,lr = 0.01):
        self.lr = lr
    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

