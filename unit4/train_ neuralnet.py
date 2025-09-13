# 相对导入：
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dataset.mnist import load_mnist
from unit4.two_layer_net import TwoLayerNet
(x_train, t_train),( x_test, t_test ) = load_mnist(normalize=True, one_hot_label=True)
#(x_train, t_train),(x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=True)


train_loss_list = []
# 超参 
iters_num = 1000  # 迭代次数，每一次都会做一次minibatch
mini_batch_size = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1


network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #计算梯度
    gradient = network.numerical_gradient(x_batch, t_batch)
    
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * gradient[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    #train_loss = network.loss(x_batch, t_train)
    print(f"第 {i} 次 推理结束...")
  