# 相对导入：
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

def getdata():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)   
    return x_test, t_test

def init_network():
    with open("dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

x,t = getdata()
network = init_network()
#print("Keys in network:", list(network.keys()))

# 开始学习预测
batch_size = 100
accuracy_cnt = 0 
for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch,axis=1) # 获取概率最高的元素的索引
    print(f"第{ i} 批精确度：{p}")
    print(f"测试精确度数据：{t[i:i+batch_size]}")
    accuracy_cnt += np.sum( p == t[i:i+batch_size])
    #print(f"总共精确度：{accuracy_cnt}")
print(f":::{t}")
print("Accuracy:", str(float(accuracy_cnt) / len(x)))