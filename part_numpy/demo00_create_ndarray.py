import numpy as np

def create_ndarray():
    """array函数创建ndarray"""
    print("=" * 10, "创建ndarray", "=" * 10)
    arr = np.array([1, 2, 3, 4, 5])
    print(arr)
    print()
def create_ndarray2():
    """asarray()函数创建ndarray，与array()函数类似，但asarray()函数不会复制数据"""
    print("=" * 10, "asarray函数创建ndarray", "=" * 10)
    arr = np.asarray([1, 2, 3, 4, 5])
    print(arr)
    print()

def create_ndarray3():
    """frombuffer()函数创建ndarray，与array()函数类似，但frombuffer()函数不会复制数据"""
    print("=" * 10, "frombuffer函数创建ndarray", "=" * 10)
    arr = np.frombuffer(bytes([1, 2, 3, 4, 5]), dtype=np.uint8) # 不能传入列表
    print(arr)
    print()

def create_ndarray4():
    """fromfile()函数创建ndarray，与array()函数类似，但fromfile()函数不会复制数据"""
    print("=" * 10, "fromfile函数创建ndarray", "=" * 10)
    file_path = "/Users/luyuan/neulife/pyproject/deepLearning-Basic/part_numpy/array_data.bin"
    arr = np.fromfile(file_path,dtype=np.int32,sep=' ')
    print(arr)
    print()


def create_ndarray5():
    """fromstring()函数创建ndarray，与array()函数类似，但fromstring()函数不会复制数据，而是将字符串转换为ndarray"""
    print("=" * 10, "fromstring函数创建ndarray", "=" * 10)
    arr = np.fromstring("1 2 3 4 5 6 7 8 9 10", dtype=np.int32, sep=' ')
    print(arr)
    print()

def create_ndarray6():
    """fromiter()函数创建ndarray，与array()函数类似，但fromiter()函数不会复制数据，而是将迭代器转换为ndarray"""
    print("=" * 10, "fromiter函数创建ndarray", "=" * 10)
    arr = np.fromiter(range(10), dtype=np.int32)
    print(arr)
    print()

def create_ndarray7():
    """fromfunction()函数创建ndarray，与array()函数类似，但fromfunction()函数不会复制数据，而是将函数转换为ndarray"""
    print("=" * 10, "fromfunction函数创建ndarray", "=" * 10)
    arr = np.fromfunction(lambda i, j: i + j, (3, 4), dtype=np.int32)
    print(arr)
    print()

def create_ndarray8():
    """ones()函数创建ndarray，与array()函数类似，但ones()函数不会复制数据，而是将函数转换为ndarray"""
    print("=" * 10, "ones函数创建ndarray", "=" * 10)
    arr = np.ones((3, 4), dtype=np.int32)
    print(arr)
    print()

def create_ndarray19():
    """ones_like()函数创建ndarray，与array()函数类似，但ones_like()函数不会复制数据，而是将函数转换为ndarray"""
    print("=" * 10, "ones_like函数创建ndarray", "=" * 10)
    raw_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr = np.ones_like(raw_arr, dtype=np.int32)
    print(arr)
    print()

def create_ndarray10():
    """zeros()函数创建ndarray，与array()函数类似，但zeros()函数不会复制数据，而是将函数转换为ndarray"""
    print("=" * 10, "zeros函数创建ndarray", "=" * 10)
    arr = np.zeros((3, 4), dtype=np.int32)
    print(arr)
    print()

def create_ndarray11():
    """zeros_like()函数创建ndarray，与array()函数类似，但zeros_like()函数不会复制数据，而是将函数转换为ndarray"""
    print("=" * 10, "zeros_like函数创建ndarray", "=" * 10)
    raw_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr = np.zeros_like(raw_arr, dtype=np.int32)
    print(raw_arr)
    print("-" * 10)
    print(arr)
    print()

def create_ndarray12():
    """empty()函数创建ndarray，与array()函数类似，但empty()函数不会复制数据，而是将函数转换为ndarray"""
    # 通过分配新内存创建数组VI区别于 ones 和 zeros不填充任何值
    print("=" * 10, "empty函数创建ndarray", "=" * 10)
    arr = np.empty((3, 4), dtype=np.int32)
    print(arr)
    print()

def create_ndarray13():
    """empty_like()函数创建ndarray，与array()函数类似，但empty_like()函数不会复制数据，而是将函数转换为ndarray"""
    # 创建一个与 raw_arr 形状相同，但未初始化的数组。其内容是垃圾值，通常为内存的值。
    print("=" * 10, "empty_like函数创建ndarray", "=" * 10)
    raw_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr = np.empty_like(raw_arr, dtype=np.int32)
    print(raw_arr)
    print("-" * 10)
    print(arr)
    print()

def create_ndarray15():
    """full()函数创建ndarray，与array()函数类似，但full()函数不会复制数据，而是将函数转换为ndarray"""
    print("=" * 10, "full函数创建ndarray", "=" * 10)
    arr = np.full((3, 4), 1, dtype=np.int32)
    print(arr)
    print()

def create_ndarray16():
    """full_like()函数创建ndarray，与array()函数类似，但full_like()函数不会复制数据，而是将函数转换为ndarray"""
    print("=" * 10, "full_like函数创建ndarray", "=" * 10)
    raw_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr = np.full_like(raw_arr, 1, dtype=np.int32)
    print(raw_arr)
    print("-" * 10)
    print(arr)
    print()

def create_ndarray14():
    """eye()函数创建ndarray，与array()函数类似，但eye()函数不会复制数据，而是将函数转换为ndarray"""
    print("=" * 10, "eye函数创建ndarray", "=" * 10)
    arr = np.eye(3, dtype=np.int32) # 创建一个3x3的单位矩阵
    print(arr)
    print()

def create_ndarray17():
    arr = np.identity(3, dtype=np.int32) # 创建一个3x3的单位矩阵
    print(arr)
    print()

if __name__ == "__main__":
    create_ndarray17()