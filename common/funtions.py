import numpy as np
from .gradient import numerical_gradient

def step_function(x):
    return np.array(x>0,dtype=np.int)


def sigmoid(x):
    return 1/(1+np.exp(-x))

# 激活函数的导数
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return np.maximum(0,x)

def identity_function(x):
    return x

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y


def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error2(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size
    

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
    return x 
