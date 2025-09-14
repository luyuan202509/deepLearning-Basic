''' 训练神经网络-每个epoch下评估损失值'''
# 相对导入：
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dataset.mnist import load_mnist
from unit4.two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

(x_train, t_train),( x_test, t_test ) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


# 超参 
iters_num = 10000  # 迭代次数，每一次都会做一次minibatch
train_size = x_train.shape[0]
batch_size = 100 
learning_rate = 0.1



train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch  = max(train_size / batch_size, 1)

print(f'训练开始始始始始始始始始始:{iter_per_epoch}')


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
    
    if i % int(iter_per_epoch)== 0:
        train_acc= network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

        
    #print(f"第 {i} 次 推理结束...")
    # 监控训练过程
    if i % 100 == 0:
        print(f"第 {i} 次迭代，损失值: {loss}")



# 绘制训练损失曲线

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


# 打印一些关键点的损失值
print(f"初始损失值: {train_loss_list[0]}")
print(f"最终损失值: {train_loss_list[-1]}")
print(f"最小损失值: {min(train_loss_list)}")