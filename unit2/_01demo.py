import numpy as np

def AND2(x1,x2):
    w1,w2,theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2 
    if tmp <= theta:
        return 0
    else:
        return 1

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1,x2):
    s1 = AND(x1,x2)
    s2 = OR(x1,x2)
    return AND(s1,s2)

def step_function(x):
    return np.array(x>0,dtype=np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)


def main():
    x = np.array([-1.0,0.5,1.0,2.0])
    y = sigmoid(x)
    print(y)
  
if __name__ == "__main__":
    main()