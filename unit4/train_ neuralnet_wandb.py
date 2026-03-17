import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt
import wandb


# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="cloudlab",
    # Set the wandb project where this run will be logged.
    project="minist_demo",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.1,
        "architecture": "MLP",
        "dataset": "MNIST",
        "epochs": 10000,
    },
)




(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True,one_hot_label=True)

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch 的重复次数
iter_per_epoch = max(train_size / batch_size, 1)
print(f"22222:{iter_per_epoch} ")

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

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

        run.log({"train_acc": train_acc, "test_acc": test_acc})

run.finish()


#plt.plot(train_acc_list, label="train acc")
#plt.plot(test_acc_list, label="test acc")
#plt.xlabel("epochs")
#plt.ylabel("accuracy")
#plt.legend()
#plt.show()

#plt.plot(train_loss_list)
#plt.show()