"""
# 通用函数:快速的元素级数组函数
通用函数( u func)是一种对 ndarray 中的数据执行元素级运算的函数。
你可以将其看作 简单函数的快速向量化封装器,这些函数接收一个或多个标量值,并生成一个或多个标量值的结果。
"""
import numpy as np
def getRng():
    """获取随机数生成器"""
    print("=" * 10, "获取随机数生成器", "=" * 10)
    rng = np.random.default_rng(seed=12345)
    #print(rng)
    return rng 

# ==============================================================================
# == 一元通用函数 
# ==============================================================================
"""
abs, fabs 逐元素地计算整数、浮点数或复数的绝对值 
sqrt 计算各元素的平方根,等价于 arr ** 0.5 
square 计算各元素的平方,等价于 arr ** 2 
exp 计算各元素的自然指数 e x  
log、log10、log2、 log1p  分别为自然对数(底数为 e)、底数为 10 的对数、底数为 2 的对数 和 log(1+x)  
sign 计算各元素的符号:1(正数)、0(零)、-1(负数) 
ceil 计算各元素的最大整数值,即大于等于该值的最小整数 
floor 计算各元素的最小整数值,即小于等于该值的最大整数 
rint 将各元素值四舍五入到最接近的整数,保留 dtype 
modf 将数组的小数部分和整数部分以两个独立数组的形式返回 
isnan 返回表示“哪些值是 NaN(不是一个数值)”的布尔型数组

isfinite、isinf 分别返回表示“哪些元素是有限的(非 inf,非 NaN)”和“哪些  元素是无限的”的布尔型数组  
cos、 cosh、 sin、 sinh、tan、tanh  正则三角函数和双曲三角函数
arccos、 arccosh、 arcsin、 arcsinh、 arctan、arctanh  反三角函数
logical_not 对数组的元素按位取反,等价于 ~arr

"""

def unary_func():
    """一元通用函数"""
    print("=" * 10, "一元通用函数", "=" * 10)
    arr = np.random.standard_normal(size=(3, 4))
    print("arr:\n", arr)
    print("abs(arr):\n", np.abs(arr))
    print("sqrt(arr):\n", np.sqrt(arr))
    print("square(arr):\n", np.square(arr))
    print("exp(arr):\n", np.exp(arr))
    print("log(arr):\n", np.log(arr))
    print("log10(arr):\n", np.log10(arr))
    print("log2(arr):\n", np.log2(arr))
    print("log1p(arr):\n", np.log1p(arr))
    print("sign(arr):\n", np.sign(arr))
    print("ceil(arr):\n", np.ceil(arr))
    print("floor(arr):\n", np.floor(arr))
    print("rint(arr):\n", np.rint(arr))
# ==============================================================================
# == 二元通用函数 
# ==============================================================================
"""
add 在数组中添加相应的元素 
subtract 从第一个数组的元素中减去第二个数组的元素 
multiply 将数组中对应的元素相乘 
divide、floor_divide 除或整除(截断余数) 
power 第一个数组中的各元素以第二个数组中的相应元素做幂次方 
maximum、fmax 逐个元素计算最大值,fmax 忽略 NaN 
minimum、fmin 逐个元素计算最小值,fmin 忽略 NaN 
mod 元素级的求模计算(除法的余数) 
copysign 将第二个数组中元素值的符号复制给第一个数组中的元素
greater、 greater_equal、 less、less_equal, equal、 not_equal  执行元素级的比较运算,返回布尔型数组。相当于中缀运

logical_and 执行元素级的逻辑与(&)运算 
logical_or 执行元素级的逻辑或(|)运算 
logical_xor 执行元素级的逻辑异或(^)运算
"""

def binary_func():
    """二元通用函数"""
    print("=" * 10, "二元通用函数", "=" * 10)
    arr1 = np.random.standard_normal(size=(3, 4))
    arr2 = np.random.standard_normal(size=(3, 4))
    print("arr1:\n", arr1)
    print("arr2:\n", arr2)
    print("add(arr1, arr2):\n", np.add(arr1, arr2))
    print("subtract(arr1, arr2):\n", np.subtract(arr1, arr2))

if __name__ == "__main__":
    binary_func()