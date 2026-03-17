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
    if a.ndim == 2:
        # 对每个样本（每一行）减去该行的最大值，防止溢出
        a = a - np.max(a, axis=1, keepdims=True)
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
        y = exp_a / sum_exp_a
    else:
        # 单样本情况
        a = a - np.max(a)
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
    return y

def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error2(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

def cross_entropy_error(y,t):
    """交叉熵误差
    y: 预测概率 (batch, num_classes) 或 (num_classes,)
    t: 教师标签 (batch,) 的类别索引，或 (batch, num_classes) 的 one-hot
    """
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = np.array(t).reshape(1, -1)

    batch_size = y.shape[0]

    # one-hot -> label index
    if t.ndim != 1 and t.size == y.size:
        t = np.argmax(t, axis=1)

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
    return x 
