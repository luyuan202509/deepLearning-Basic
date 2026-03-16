import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dataset.mnist import load_mnist
from common.funtions import sigmoid,cross_entropy_error,softmax
from common.optimizer import SGD
from common.gradient import numerical_gradient
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True,one_hot_label=True)

train_loss_list = []

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取 mini_batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    #grads = network.numerical_gradient(x_batch, t_batch)
    grads = network.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grads[key]

    
    # 记录学习过程
    train_loss = network.loss(x_batch, t_batch)
    train_loss_list.append(train_loss)

plt.plot(train_loss_list)
plt.show()