import numpy as np

def identity_function(x):
    """恒等函数"""
    return x
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
def softmax2(x):
   c = np.max(x)
   exp_a = np.exp(x-c)
   sum_exp_a = np.sum(exp_a)
   y = exp_a / sum_exp_a
   return y


def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)


def main1():
    x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6]])
    y = softmax(x)
    print(y)

def main2():
    x = np.array([0.1, 0.8, 0.1])
    y = softmax(x)
    print(y)

if __name__ == "__main__":
    main1()
    main2()