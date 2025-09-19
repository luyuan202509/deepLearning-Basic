# 正确计算项目根目录
import os,sys
from tkinter import N 
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
review_dir = os.path.dirname(current_dir)  # review目录
project_root = os.path.dirname(review_dir)  # 项目根目录
sys.path.append(project_root)

class MulLayer:
    """正向传播和方向传播的 乘法层 """
    def __init__(self,) -> None:
        self.x = None
        self.y = None
    
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self,dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy

class AddLayer:
    """ 加法器层正向，反向传播 """
    def __init__(self,):
        self.x = None
        self.y = None
    
    def forward(self,x,y):
        return x + y
    
    def backward(self,dout):
        return dout,dout
