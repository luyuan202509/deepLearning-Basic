import numpy as np
import sys, os
# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from dataset.mnist import load_mnist
import pickle
from common.functions import softmax,sigmoid

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
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

def main():
    network = init_network()
    x, t = get_data()
   
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)  # 获取概率最高的元素的索引
        if p == t[i]:
            accuracy_cnt += 1
        
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

def main2():
    network = init_network()
    x, t = get_data()
   
    batch_size = 100 
    accuracy_cnt = 0

    for i in range(0, len(x),batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
        
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

if __name__ == "__main__":
    #main()
    main2()