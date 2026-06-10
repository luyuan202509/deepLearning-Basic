import torch
import numpy as np


def tensor_add():
    """Tensor 加法"""
    print("=" * 10, "Tensor 加法", "=" * 10)
    a = torch.tensor([[1, 2, 3], [4, 5, 6]],dtype=torch.float32)
    b = torch.tensor([[1, 2, 3], [4, 5, 6]],dtype=torch.float32)

    #c = a + b 直接加法
    c = torch.empty_like(a)
    c = torch.add(a, b, out=c)
    #c = torch.add(a, b)
        
    print("c:\n", c)
    print()
   
def tensor_add2():
    """Tensor 加法"""
    print("=" * 10, "Tensor 加法", "=" * 10)
    a = torch.tensor([[1, 2, 3], [4, 5, 6]],dtype=torch.float32)
    b = torch.tensor([[1, 2, 3], [4, 5, 6]],dtype=torch.float32)

    #a += b
    a.add_(b) 
    print("a:\n", a)
        
    #print("c:\n", c)
    print()

def tensor_other():
    """Tensor 其他操作"""
    print("=" * 10, "Tensor 其他操作", "=" * 10)
    a = torch.tensor([[1, 2, 3], [4, 5, 6]],dtype=torch.float32)
    b = torch.tensor([[4,5,6], [7,8,9]],dtype=torch.float32)
    
    c = a * b
    d = a / b
    e = a % b
    f =  a // b 
    print("c:\n", c)
    print("d:\n", d)
    print("e:\n", e)
    print("f:\n", f)
    print()
   
if __name__ == "__main__":
    tensor_other()

    # https://www.bilibili.com/video/BV1iv41117Zg?t=553.0&p=5
