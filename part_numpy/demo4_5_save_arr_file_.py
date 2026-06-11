"""
NumPy 能够读写磁盘上的文本数据或二进制格式的数据。
"""
import numpy as np

def save_arr_file():
    """保存数组到文件"""
    arr = np.arange(10)
    np.save("some_arr.npy",arr)
    print("保存数组到文件 some_arr.npy 成功")

def load_arr_file():
    """加载数组到文件"""
    arr = np.load("some_arr.npy")
    print(arr.dtype)
    #print("arr:\n", arr)
    print("加载数组 from some_arr.npy 成功")


if __name__ == "__main__":
    load_arr_file()