# 相对导入：
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np

def function(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.minorticks_on()
plt.plot(x,y)
plt.show()