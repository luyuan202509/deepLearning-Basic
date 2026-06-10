import torch
import numpy as np


def demo_tensor_creation():
    """Tensor 对象创建与基本属性"""
    print("=" * 10, "Tensor 创建与属性", "=" * 10)

    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print("tensor:\n", tensor)
    print("shape:", tensor.shape)
    print("size():", tensor.size())
    print("dtype:", tensor.dtype)
    print()


def demo_tensor_generation():
    """Tensor 数据生成: 常量 / 随机分布"""
    print("=" * 10, "Tensor 数据生成", "=" * 10)

    # 常量生成
    ones_tensor = torch.ones([2, 3],dtype=torch.float32)
    print("ones:\n", ones_tensor)

    zeros_tensor = torch.zeros([2, 3],dtype=torch.float32)
    print("zeros:\n", zeros_tensor)

    # 随机分布生成
    randn_tensor = torch.randn([2, 3])          # 标准正态分布
    print("randn (正态):\n", randn_tensor)

    rand_tensor = torch.rand([2, 3])            # [0, 1) 均匀分布
    print("rand (均匀):\n", rand_tensor)

    randint_tensor = torch.randint(1, 10, [2, 3])  # 整数随机
    print("randint (整数):\n", randint_tensor)

    # 按已有 tensor 形状生成同型随机 tensor
    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = torch.rand_like(a, dtype=torch.float32)
    print("原 tensor:\n", a)
    print("rand_like:\n", b)
    print()


def demo_tensor_reshape():
    """Tensor 形状变换: view"""
    print("=" * 10, "Tensor 形状变换", "=" * 10)

    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("原始 (3×3):\n", a)

    # 展平为 1D
    flat = a.view(9)
    print("view(9) 展平:\n", flat)

    # 恢复为 3×3
    restored = flat.view(3, 3)
    print("view(3,3) 恢复:\n", restored)

    # 2×3 → 3×2 变换
    e = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("原始 (2×3):\n", e)
    f = e.view(3, 2)
    print("view(3,2):\n", f)
    print()
 
def demo_tensor_getitem():

    """Tensor 获取元素"""
    print("=" * 10, "Tensor 获取元素", "=" * 10)

    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=torch.float32)
    print("tensor:\n", tensor)
    print("tensor[0]:\n", tensor[0])
    print("tensor[1]:\n", tensor[1])
    print("tensor[2]:\n", tensor[2])
    print("="*5)
    print("tensor[0,0]:\n", tensor[0,0].item())
    print("tensor[0,1]:\n", tensor[0,1].item())
    print("tensor[0,2]:\n", tensor[0,2].item())

def demo_tensor_numpy():

    """Tensor 转换为 NumPy 数组"""
    print("=" * 10, "Tensor 转换为 NumPy 数组", "=" * 10)

    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=torch.float32)
    print("tensor:\n", tensor)
    print("tensor.numpy():\n", tensor.numpy())
    print()
    """NumPy 数组转换为 Tensor"""
    numpy_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("numpy_array:\n", numpy_array)
    print("torch.from_numpy(numpy_array):\n", torch.from_numpy(numpy_array)) # 关键
    print()


if __name__ == "__main__":
    # demo_tensor_creation()
    # demo_tensor_generation()
    # demo_tensor_reshape()
    demo_tensor_numpy()