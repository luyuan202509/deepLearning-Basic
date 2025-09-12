''' 常用公共函数 '''
import numpy as np

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