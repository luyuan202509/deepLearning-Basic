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
    with open("dataset/mnist.pkl", 'rb') as f:
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

x,y = getdata()
network = init_network()
print("Keys in network:", list(network.keys()))