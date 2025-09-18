import pickle
import os,sys

# 正确计算项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
review_dir = os.path.dirname(current_dir)  # review目录
project_root = os.path.dirname(review_dir)  # 项目根目录
sys.path.append(project_root)
from common.functions import sigmoid,softmax

import numpy as np
from dataset.mnist import load_mnist
import pickle
def getdata():
    (x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        return network

# 向前传播
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


x, t = getdata()
network = init_network()
accuracy_cnt = 0

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
