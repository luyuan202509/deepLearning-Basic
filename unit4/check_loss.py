import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
from common.funtions import numerical_gradient,gradient_descent

def function_1(x):
    return x[0]**2 + x[1]**2


def function_2(x):
    return x[0]**2 + (x[1])**2



if __name__ == '__main__':
    init_x = np.array([-3.0,4.0])
    x =  gradient_descent(function_2,init_x,lr = 0.1,step_num = 100)
    print(x)


