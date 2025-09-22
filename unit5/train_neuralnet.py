import sys,os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dataset.mnist import load_mnist
from unit5.two_layer_net import TwoLayerNet
 
(x_train,t_train),(x_test,t_test) = load_mnist(normalize = True,one_hot_label = True)

network = TwoLayerNet(input_size = 784,hidden_size = 50,output_size = 10)

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100 
learn_rate = 0.009
train_loss_list = []
train_acc_list = []
test_acc_list = []

def recordLoss(i,loss):
    with open("unit5/loss.txt", "a") as f:
        f.write( f"{i} : " + str(loss) + "\n")


iter_per_epoch = max(train_size / batch_size,1)
for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
   
   # 误差反向传播
    grad = network.gradient(x_batch, t_batch)
    
    #更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learn_rate * grad[key]
    
        # 计算损失
        loss  = network.loss(x_batch, t_batch)

        #记录损失
        recordLoss(i,loss)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
   # print(f"第 {i} 次 推理结束...");

#print("损失：",train_loss_list)


