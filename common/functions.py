''' 常用公共函数 '''
import numpy as np

from test import numerical_diff

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 越迁函数
def step_function(x):
    return np.array(x > 0, dtype=np.int)


''' 输出问题 '''
# 恒等函数
def identity_function (x):
   return x 
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y 

'''损失函数'''
# 均方误差
def mean_squared_error(y,t):
    return 0.5 * np.sum(y-t**2)
# 交叉熵误差
def cross_entropy_error(y,t):
    delt  = 1e-7
    return -np.sum(t * np.log(y-delt))

# mini-batch版交叉熵误差的实现,监督数据只有一个正确结果或者有多个
def cross_entropy_error(y,t,one_hot_label):
    delt  = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    if not one_hot_label:
        return -np.sum(y[np.log([y,batch_size]),t]+ 1e-7) / batch_size
    else:
        return -np.sum(t * np.log(y-delt)) / batch_size

# 偏导数计算
def numerical_gradient(f,x):
    h= 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
    
        #f(x+h) 计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f[x-h] 计算
        x[idx] = tmp_val -h 
        fxh2 = f(x)
        
        grad[idx] = (fxh1-fxh2) / 2*h 
        x[idx]= tmp_val

    return grad 


# 梯度下降法
def gradient_descent(f,init_x,lr = 0.01,step_num =100):
    x = init_x
    for i in range(step_num):
        grad = numerical_diff(f,x)
        x -= lr * grad
        