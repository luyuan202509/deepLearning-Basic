import numpy as np  


def demo_numpy_creation():
    """NumPy 数组切片"""
    print("=" * 10, "NumPy 数组切片", "=" * 10)
   # arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr = np.arange(9).reshape(3, 3)
    #arr = np.arange(9).reshape(3,3).view()
    print("arr:\n", arr)

    print("arr[0]:\n", arr[0])
    print("arr[:1]:\n", arr[:2])


if __name__ == "__main__":
    demo_numpy_creation()