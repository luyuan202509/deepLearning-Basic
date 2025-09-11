from math import pi
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from  dataset.mnist import  load_mnist
import numpy as np
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train),(x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=True)

img = x_test[0]
label = t_test[0]
print(label)

print(img.shape)
img = img.reshape(28,28)
print(img.shape)

img_show(img)
