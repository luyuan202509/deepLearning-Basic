import numpy as np
import matplotlib.pyplot as plt
import sys, os
# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from common.functions import sigmoid

x = np.random.rand(1000, 100) # 随机1000 个数据 

node_num=100          # 每个隐藏层节点（神经元）数量
hidden_layersize = 5  # 隐藏层 5
activations = {}      # 激活值


for i in range(hidden_layersize):
    if i != 0:
        x = activations[i-1]
    
    #w = np.random.randn(node_num, node_num) * 1 
    w = np.random.randn(node_num, node_num) * 0.01

    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z

for i,a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0,1))

plt.show()


