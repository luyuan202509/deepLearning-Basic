import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train),(x_test, t_test) = load_mnist(flatten=True,normalize=False)


batch_size = 10
train_size = x_train.shape[0]
batch_mask = np.random.choice(train_size, batch_size,replace=False)

print(x_train.shape)
print(train_size) # 60000
print(batch_mask)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_batch.shape)
print(t_batch.shape)