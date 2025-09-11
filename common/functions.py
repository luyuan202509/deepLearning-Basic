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
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y 