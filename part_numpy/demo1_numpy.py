import numpy as np

def slice_numpy():
    """一维切片"""
    print("=" * 10, "一维切片", "=" * 10)
    arr = np.arange(9)
    print("arr:\n", arr)
    print("arr[2:5]:\n", arr[2:5])
    print("arr[:5]:\n", arr[:5])
    print("arr[5:]:\n", arr[5:])
    print("arr[:]:\n", arr[:])
    print("="*5)
    print("arr[::2]:\n", arr[::2]) # 步长为2
    print("arr[1::2]:\n", arr[1::2]) # 步长为2，从1开始
    print("arr[::-1]:\n", arr[::-1]) # 步长为-1，从后向前
    print("arr[:-3:-1]:\n", arr[:-3:-1]) # 步长为-1，从后向前，到-3为止
    print("arr[:-3:1]:\n", arr[:-3:1]) # 步长为1，从后向前，到-3为止
    print("arr[:-3:-1]:\n", arr[:-3:-1]) # 步长为-1，从后向前，到-3为止

def slice_numpy2():
    """二维切片，在二维数组中，各索引位置上的元素不再是标量，而是一个一维数组"""
    print("=" * 10, "切片", "=" * 10)
    arr2d= np.arange(9).reshape(3, 3)
    print("arr2d:\n", arr2d)

    # 单个元素访问
    # print("arr2d[1,1]:\n", arr2d[1,1])
    # print("arr2d[1,1]:\n", arr2d[1][1])

    # 切片访问 
    print("arr2d[0:2]:\n", arr2d[0:2])
    print("arr2d[0:2,1:3]:\n", arr2d[0:2,1:3])


def bool_index():
    """布尔索引"""
    names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
    data = np.random.randint(0, 10, (7, 4))
    print("names:\n", names)
    print("data:\n", data)

    # 选出data中名字为Bob的一行
    # name_bool = names == 'Bob'
    # print("arr_bool:\n", name_bool)
    # print("data[name_bool]:\n", data[name_bool])
    
    # 选出data中名字为Bob之外的一行
    #name_bool = names != 'Bob'
    # print("arr_bool:\n", name_bool)
    # print("data[name_bool]:\n", data[name_bool])

    # 选出data中名字为Bob之外的一行
    #name_bool = names == 'Bob'
    #name_bool_not = ~name_bool  # ~ 运算符 反转布尔数组
    # print("name_bool:\n", name_bool)
    # print("name_bool_not:\n", name_bool_not)
    # print("data[name_bool_not]:\n", data[name_bool_not]) # 布尔索引 选出data中名字为Bob之外的一行
    

    # 选出data中名字为Bob，和 Joe之外的行 （运算符 &（与） |（或） ）
    # mask = (names == 'Bob') | (names == 'Will')
    # print("mask:\n", mask)
    # print("data[mask]:\n", data[mask])

    
    # 通过一维布尔数组设置正行或整列的值 （布尔索引） 把names中不是Joe的行设置为7
    mask = names != 'Joe'
    data[mask] = 7
    print("mask:\n", mask)
    print("data:\n", data)

def fancy_index():
    """花式索引"""
    arr = np.zeros((8, 4))
    #print("arr:\n", arr)

    for i in range(8):
        arr[i] = i
    print("arr:\n", arr)
    
    # index_arr = [4, 3, 0, 6] # 花式索引 选出arr的第4，3，0，6行
    # print("arr[index_arr]:\n", arr[index_arr])

    # index_arr = [-3, -5, -7] # 花式索引 选出arr的倒数第3，5，7行
    # print("arr[index_arr]:\n", arr[index_arr])

    mask = [[1,5,7,2],[0, 3, 1, 2]] # 花式索引 选出arr的第1，5，7，2行和第0，3，1，2列
    print("arr[index_arr]:\n", arr[mask])

def fancy_index2():
    """花式索引"""
    arr = np.arange(32).reshape(8, 4)
    print("arr:\n", arr)
    # print("arr[mask]:\n", arr[[1,5,7,2],[0,3,1,2]]) #花式索引 选出arr的第1，5，7，2行和第0，3，1，2列

    mask_row = [1,5,7,2]
    mask_col = [0,3,1,2]
    print("arr[mask]:\n", arr[mask_row, mask_col]) #花式索引 选出arr的第1，5，7，2行和第0，3，1，2列

def matrixOperate():
    """矩阵操作: 矩阵转置与矩阵乘法"""
    arr = np.arange(15).reshape(3, 5)
    print("arr:\n", arr)


    # 矩阵转置
    arr_T = arr.T
    print("arr.T:\n", arr_T)
    # 矩阵乘法 dot() 方法
    print("arr.dot(arr_T):\n", arr.dot(arr_T))
    # 矩阵乘法2 @ 符号运算符
    print("arr.dot(arr.T):\n", arr @ arr_T)

if __name__ == "__main__":
    matrixOperate()