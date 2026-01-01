import os,sys
#sys.path.append('..')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from common.util import im2col, col2im
from collections import OrderedDict
from common.layers import Convolution,Relu,Pooling,Affine,SoftmaxWithLoss

class Convolution:
    def __init__(self,W,b,stride=1,pad=0):
        """初始化参数，W:权重,b:偏置,stride:步幅,pad:填充，权重就是滤波器，b:偏置"""
        self.W = W  # 权重也是滤波器
        self.b = b  # 偏置
        self.stride = stride
        self.pad = pad
    
    def forward(self,x): # x:输入数据 图片大小为(N,C,H,W)
        FN,C,FH,FW = self.W.shape # FN:滤波器数量,C:输入通道数,FH:滤波器高度,FW:滤波器宽度
        N,C,H,W = x.shape # N:批量大小,C:输入通道数,H:输入高度,W:输入宽度
        out_h = int((H +2*self.pad -FH)/self.stride +1) 
        out_w = int((W +2*self.pad -FW)/self.stride +1) 
        # 将输入数据展开为行
        col = im2col(x,FH,FW,self.stride,self.pad) # 将输入数据展开为行
        col_W = self.W.reshape(FN,-1).T # 将权重展开为列
        out = np.dot(col,col_W) + self.b

        out = out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)
        return out

class Pooling:
    def __init__(self,pool_h,pool_w,stride=1,pad =0):
        """初始化参数，权重就是滤波器，b:偏置"""
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
    
    def forward(self,x):
        N,C,H,W = x.shape
        out_h = int((H - self.pool_h)/self.stride +1)
        out_w = int((W - self.pool_w)/self.stride +1)
        
        # 展开 1
        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w) # 将每个池化窗口展开为行
        
        # max池化 2 
        out = np.max(col,axis=1)
        
        # 还原数据形状 将输出数据展开为(N,C,out_h,out_w) 3 
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        
        return out 



class SimpleConvNet:
    """简单的卷积神经网络"""
    def __init__(self,input_dim=(1,28,28),
                      conv_param = {'filter_num':30,'filter_size':5,'pad':0,'stride':1},
                      hidden_size=100,output_size=10,weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size + 2*filter_pad - filter_size ) /filter_stride + 1 
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num,input_dim[0],filter_size,filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size,hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],self.params['b1'],conv_param['stride'],conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2,pool_w=2,stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'],self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],self.params['b3'])

        self.lastLayer = SoftmaxWithLoss()


    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
    
    def gradient(self,x,t):
        """求梯度（误差反向传播法"""
        # forward
        self.loss(x,t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
        

