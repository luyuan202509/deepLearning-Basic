import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dataset.mnist import load_mnist
from unit5.two_layer_net import TwoLayerNet
# 读入数据
(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


x_batch = x_train[:3]
t_batch = t_train[:3]
# 正向转播梯度下降
grads_numerical =  network.numerical_gradient(x_batch,t_batch)
# 反向传播梯度下降
grads_backprop = network.gradient(x_batch,t_batch)

for key in grads_numerical.keys():
    diff = np.average(np.abs(grads_numerical[key] - grads_backprop[key]))
    print(f"对比两个版本{key}:{str(diff)}")