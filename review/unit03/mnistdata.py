import os,sys 
# 相对导入：
import sys,os
# 正确计算项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
review_dir = os.path.dirname(current_dir)  # review目录
project_root = os.path.dirname(review_dir)  # 项目根目录
sys.path.append(project_root)
# 从 dataset 目录导入
print(current_dir)
print(review_dir)
print(project_root)


from dataset.mnist import load_mnist
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)   

print(x_train.shape)
