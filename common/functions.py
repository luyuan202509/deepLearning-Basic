''' 常用公共函数 '''
import numpy as np

def identity_function(x):
    return x
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 激活函数的导数
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
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

# 版本1
'''
def cross_entropy_error(y,t,one_hot_label):
    delt  = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    if one_hot_label:
        return -np.sum(t * np.log(y-delt)) / batch_size
    else:
        return -np.sum(y[np.log([y,batch_size]),t]+ 1e-7) / batch_size

'''

#版本二
def cross_entropy_error(y, t): 
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 偏导数计算
def numerical_gradient1(f,x):
    ''' 一纬数组 '''
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
        
        grad[idx] = (fxh1-fxh2) / (2*h) 
        x[idx]= tmp_val
    return grad 

def numerical_gradient(f, x):
    '''多维数组 '''
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值
        it.iternext()   
    return grad



# 梯度下降法
def gradient_descent(f,init_x,lr = 0.01,step_num =100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x) # 需要实现数值梯度计算
        x -= lr * grad
    return x
        