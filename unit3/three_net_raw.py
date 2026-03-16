import numpy as np
import sys,os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.funtions import sigmoid,identity_function

X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
W3 = np.array([[0.1,0.3],[0.2,0.4]])
B1 = np.array([0.1,0.2,0.3])
B2 = np.array([0.1,0.2])
B3 = np.array([0.1,0.2])
 
A1 = np.dot(X,W1) + B1  #(1,3)
z1 = sigmoid(A1)

A2= np.dot(z1,W2) + B2
Z2 = sigmoid(A2)

A3 = np.dot(Z2,W3) + B3
y = identity_function(A3)
print(y)

