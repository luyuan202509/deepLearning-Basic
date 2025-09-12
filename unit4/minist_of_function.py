'''
请用梯度法求 函数 f(x0+x1) = x0^2 + x1^2 的最小值
'''

# 相对导入：
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from common.functions import gradient_descent 
import math 
    

def function_2(x):
    return np.sum(x**2)

init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x,lr = 0.1,step_num =100))




