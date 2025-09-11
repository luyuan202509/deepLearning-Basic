''' 常用公共函数 '''
import numpy as np

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 越迁函数
def step_function(x):
    return np.array(x > 0, dtype=np.int)

