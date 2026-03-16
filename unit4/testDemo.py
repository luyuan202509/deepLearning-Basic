import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from common.funtions import sigmoid,identity_function,mean_squared_error,cross_entropy_error

t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y),np.array(t)))
print(cross_entropy_error(np.array(y),np.array(t)))


