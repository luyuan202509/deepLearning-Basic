''' 训练神经网络'''
# 相对导入：
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dataset.mnist import load_mnist
from unit4.two_layer_net import TwoLayerNet
(x_train, t_train),( x_test, t_test ) = load_mnist(normalize=True, one_hot_label=True)
#(x_train, t_train),(x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=True)
import matplotlib.pyplot as plt

train_loss_list = []
# 超参 
iters_num = 10000  # 迭代次数，每一次都会做一次minibatch
mini_batch_size = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.09


network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #计算梯度
    gradient = network.gradient(x_batch, t_batch)
    
    # 更新参数 沿着梯度梯度下降的方向
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * gradient[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    #print(f"第 {i} 次 推理结束...")
    # 监控训练过程
    if i % 100 == 0:
        print(f"第 {i} 次迭代，损失值: {loss}")

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_loss_list)
plt.title('Training Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


# 打印一些关键点的损失值
print(f"初始损失值: {train_loss_list[0]}")
print(f"最终损失值: {train_loss_list[-1]}")
print(f"最小损失值: {min(train_loss_list)}")