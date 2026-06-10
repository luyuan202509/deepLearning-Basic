import numpy as np
""" 随机数 """
def fancy_index2():
    """花式索引"""
    arr = np.arange(32).reshape(8, 4)
    print("arr:\n", arr)
    # print("arr[mask]:\n", arr[[1,5,7,2],[0,3,1,2]]) #花式索引 选出arr的第1，5，7，2行和第0，3，1，2列

    mask_row = [1,5,7,2]
    mask_col = [0,3,1,2]
    print("arr[mask]:\n", arr[mask_row, mask_col]) #花式索引 选出arr的第1，5，7，2行和第0，3，1，2列


if __name__ == "__main__":
    fancy_index2()
    