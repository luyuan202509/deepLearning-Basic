
import numpy as np 

def SGD(params, grads, lr=0.01):
    for i in range(len(params)):
        params[i] -= lr * grads[i]