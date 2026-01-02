### 手写数字识别
import sys, os
# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir) 

from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def img_show(img):
    plt.imshow(img)
    plt.show()

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)


img = x_train[10]
label = t_train[10]
print(label) # 

print(img.shape) # (784,)
img = img.reshape(28, 28)
print(img.shape) # (28, 28)
img_show(img)
