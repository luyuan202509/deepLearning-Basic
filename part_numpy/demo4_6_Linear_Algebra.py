"""
线性代数:如矩阵乘法、矩阵分解、行列式以及其他方阵数学等
NumPy 提供了线性代数函数库 linalg,该库包含了线性代数所需的所有功能,包括:
top 矩阵乘法 
trace 计算对角线元素的和 
det 计算矩阵行列式 
eig 计算方阵的特征值和特征向量 
inv 计算方阵的逆矩阵 
pinv 计算矩阵的 Moore-Penrose 伪逆 
qr 计算 QR 分解 
svd 计算奇异值分解(SVD) 
solve 求解线性方程组 Ax = b 中的 x,其中 A 为一个方阵 
lstsq 计算 Ax = b 的最小二乘解

"""



import numpy as np
import numpy.linalg as la


def linear_algebra():
    """线性代数"""
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[1, 2], [3, 4], [5, 6]])
    print("x:\n", x)
    print("y:\n", y)

    print("x.dot(y):\n", x.dot(y))
    print("x @ y:\n", x @ y)

if __name__ == "__main__":
    linear_algebra()