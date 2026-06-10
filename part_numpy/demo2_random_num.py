import numpy as np

""" """

def random_num():
    """随机数"""
    print("=" * 10, "随机数", "=" * 10)
    arr = np.random.rand(3, 4)
    np.random.standard_normal(size=(3, 4))
    print(arr)
    print()

if __name__ == "__main__":
    random_num()