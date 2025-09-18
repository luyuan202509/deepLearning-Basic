import os,sys 
# 相对导入：
import sys,os
# 正确计算项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
review_dir = os.path.dirname(current_dir)  # review目录
project_root = os.path.dirname(review_dir)  # 项目根目录
sys.path.append(project_root)

import numpy as np
from PIL import Image
from dataset.mnist import load_mnist

def show_img(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=True)   

print(x_train.shape)

img = x_train[0]
label = t_train[0]

print(label)
print(img.shape)
img = img.reshape(28,28)
print(img.shape)
show_img(img)





