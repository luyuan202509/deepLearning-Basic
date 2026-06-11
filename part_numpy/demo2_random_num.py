import numpy as np
"""伪随机数：伪随机数是用确定性的算法计算出来的似来自[0,1]均匀分布的随机数序列，
并不真正的随机，但具有类似于随机数的统计特征，如均匀性、独立性等。也就是说，已知种子值为M，首次运行即种子值为M，
则下一次运行时，种子值仍为M，由此即可保证下一个随机数仍为N。"""

def getRng():
    """获取随机数生成器"""
    print("=" * 10, "获取随机数生成器", "=" * 10)
    rng = np.random.default_rng(seed=12345)
    #print(rng)
    return rng 


"""numpy.random 模块对 Python 内置的 random 模块补充了一些函数,用于从多种概率分布 
  中有效地生成整个样本值数组。例如,你可以用 numpy.normal 得到一个标准正态分布 的 4×4 样本数组:
 """

def random_num():
    """standard_normal()函数生成标准正态分布的随机数"""
    print("=" * 10, "standard_normal()函数生成标准正态分布的随机数", "=" * 10)
    arr = np.random.standard_normal(size=(3, 4))
    print(arr)
    print()

def random_num2():
    rng = np.random.default_rng(seed=12345)
    arr = rng.standard_normal(size=(3, 4))
    print(arr)
    print()

#===================================================================================================================
# 表 4-3:NumPy 的随机数生成器方法
"""
permutation 返回一个序列的随机排列,或返回一个随机排列的范围 
shuffle 随机打乱一个序列 
uniform 从均匀分布中抽取样本 
integers 从一个由低到高的范围抽取随机整数 
Standard_normal 从均值为 0,标准差为 1 的正态分布中抽取样本 
binomial 从二项分布中抽取样本 
normal 从正态(高斯)分布中抽取样本 
beta 从 beta 分布中抽取样本 
chisquare 从卡方分布中抽取样本 
gamma 从 gamma 分布中抽取样本 
uniform(0,1) 从 [0, 1) 范围的均匀分布中抽取样本
"""
#===================================================================================================================
def random_num3():
    """permutation 返回一个序列的随机排列,或返回一个随机排列的范围"""
    rng = getRng()
    #arr = rng.permutation(10,100) #随机种子一样，则每次生成的随机数都一样
    arr = rng.permutation(10) #随机种子一样，则每次生成的随机数都一样
    print(arr)
    print()

def random_num4():
    """shuffle 随机打乱序列"""
    rng = getRng()
    arr = np.arange(10) #随机种子一样，则每次生成的随机数都一样
    rng.shuffle(arr) #随机打乱序列
    print(arr)
    print()

def random_num5():
    """uniform 从均匀分布中抽取样本"""
    rng = getRng()
    arr = rng.uniform(0, 1, size=(3, 4))
    print(arr)
    print()

def random_num6():
    """integers 从一个由低到高的范围抽取随机整数"""
    rng = getRng()
    arr = rng.integers(0, 10, size=(3, 4))
    print(arr)
    print()

def random_num7():
    """Standard_normal 从均值为 0,标准差为 1 的正态分布中抽取样本"""
    rng = getRng()
    arr = rng.standard_normal(size=(3, 4))
    print(arr)
    print()

def random_num8():
    """binomial 从二项分布中抽取样本"""
    rng = getRng()
    arr = rng.binomial(n=10, p=0.5, size=(3, 4))
    print(arr)
    print()

def random_num9():
    """normal 从正态(高斯)分布中抽取样本"""
    rng = getRng()
    arr = rng.normal(loc=0.0, scale=1.0, size=(3, 4))
    print(arr)
    print()

def random_num10():
    """beta 从 beta 分布中抽取样本"""
    rng = getRng()
    arr = rng.beta(a=2.0, b=2.0, size=(3, 4))
    print(arr)
    print()

def random_num11():
    """chisquare 从卡方分布中抽取样本"""
    rng = getRng()
    arr = rng.chisquare(df=2, size=(3, 4))
    print(arr)
    print()

def random_num12():
    """gamma 从 gamma 分布中抽取样本"""
    rng = getRng()
    arr = rng.gamma(shape=2.0, scale=1.0, size=(3, 4))
    print(arr)
    print()

def random_num13():
    """uniform(0,1) 从 [0, 1) 范围的均匀分布中抽取样本"""
    rng = getRng()
    arr = rng.uniform(0, 1, size=(3, 4))
    print(arr)
    print()
    
if __name__ == "__main__":
    random_num12()