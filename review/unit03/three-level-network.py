''' three level network native '''

import numpy as np 

# 激活函数-segmoid
def segmoid(x):
    return 1 / (1 + np.exp(-x))

# 输出函数-恒等函数
def identity_function(x):
    return x

X  = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], 
               [0.2, 0.4, 0.6]]) 
b1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + b1
Z1 = segmoid(A1) 
print(f'z1 shape: {Z1.shape}')

W2 =np.array([[0.1, 0.4], 
             [0.2, 0.5],
             [0.3, 0.6]])
b2 = np.array([0.1, 0.2])
A2 =np.dot(Z1,W2) + b2
Z2 = segmoid(A2)

print(f'z2 shape: {Z2.shape}')


W3 = np.array([[0.1, 0.3],
               [0.2, 0.4]])
b3 = np.array([0.1, 0.2])
A3 = np.dot(Z2,W3) +b3
Z3 = segmoid(A3)

print(f'z3 shape: {Z3.shape}')


y = identity_function(A3)
print(y)






