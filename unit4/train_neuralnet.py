import numpy as np
import sys, os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from dataset.mnist import load_mnist
import pickle
#from common.functions import *
from two_layer_net import TwoLayerNet

import matplotlib.pyplot as plt




def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return (x_train, t_train), (x_test, t_test)

(x_train, t_train), (x_test, t_test)= get_data()
train_loss_list = []
train_acc_list = []
test_acc_list = []




# 超参数：
iters_num = 10000
train_size = x_train.shape[0] #60000
batch_size = 100 
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 平均每个 epoch 的重复次数 
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 获取minibatch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grads[key]
    
    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 计算每个empch的识别精度 
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc |" + str(train_acc) + ', ' + str(test_acc))


#plt.plot(train_loss_list)
#plt.show()
