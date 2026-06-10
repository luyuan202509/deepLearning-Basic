import numpy as np  


def demo_numpy_creation():
    """NumPy 数组创建与基本属性"""
    print("=" * 10, "NumPy 数组创建与基本属性", "=" * 10)

    numpy_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=np.float32)
    print("numpy_array:\n", numpy_array)
    print("shape:", numpy_array.shape)
    print("size():", numpy_array.size)
    print("dtype:", numpy_array.dtype)
    print()

    # 创建全0数组
    print("=" * 10, "创建全0数组", "=" * 10)
    zeros_array = np.zeros([2, 3],dtype=np.float32)
    print("zeros_array:\n", zeros_array)
    print("shape:", zeros_array.shape)
    print("size():", zeros_array.size)
    print("dtype:", zeros_array.dtype)
    print()
    
    # 创建全1数组
    print("=" * 10, "创建全1数组", "=" * 10)
    ones_array = np.ones([2, 3],dtype=np.float32)
    print("ones_array:\n", ones_array)
    print("shape:", ones_array.shape)
    print("size():", ones_array.size)
    print("dtype:", ones_array.dtype)
    print()

    # 创建随机数组
    print("=" * 10, "创建随机数组", "=" * 10)
    random_array = np.random.rand(2, 3)
    print("random_array:\n", random_array)
    print("shape:", random_array.shape)
    print("size():", random_array.size)
    print("dtype:", random_array.dtype)
    print()

    # 创建随机整数数组
    print("=" * 10, "创建随机整数数组", "=" * 10)
    random_int_array = np.random.randint(0, 10, [2, 3])
    print("random_int_array:\n", random_int_array)
    print("shape:", random_int_array.shape)
    print("size():", random_int_array.size)
    print("dtype:", random_int_array.dtype)
    print()

    # 创建随机浮点数数组
    print("=" * 10, "创建随机浮点数数组", "=" * 10)
    random_float_array = np.random.rand(2, 3)
    print("random_float_array:\n", random_float_array)
    print("shape:", random_float_array.shape)
    print("size():", random_float_array.size)
    print("dtype:", random_float_array.dtype)
    print()

    # 创建随机布尔数组
    print("=" * 10, "创建随机布尔数组", "=" * 10)
    random_bool_array = np.random.rand(2, 3) > 0.5
    print("random_bool_array:\n", random_bool_array)
    print("shape:", random_bool_array.shape)
    print("size():", random_bool_array.size)
    print("dtype:", random_bool_array.dtype)
    print()

    # 创建随机数组
    print("=" * 10, "创建随机数组", "=" * 10)
    random_array = np.random.rand(2, 3)
    print("random_array:\n", random_array)
    print("shape:", random_array.shape)
    print("size():", random_array.size)
    print("dtype:", random_array.dtype)
    print()

if __name__ == "__main__":
    #demo_numpy_creation()
    arr = np.arange(9)
    a = arr[2,3,4]
    print(a)