# 正确计算项目根目录
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
review_dir = os.path.dirname(current_dir)  # review目录
#project_root = os.path.dirname(review_dir)  # 项目根目录
sys.path.append(review_dir)

from unit05.layer_naive import MulLayer
apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward

apple_price = mul_apple_layer.forward(apple, apple_num)
prise = mul_tax_layer.forward(apple_price, tax)

print("apple totle price:", int(prise))

# backward
dprice = 1
dapple_price, d_apple_num = mul_apple_layer.backward(dprice)
dapple, dnumber = mul_tax_layer.backward(dapple_price)
print(f'苹果的价格梯度:{dapple},苹果数量梯度：{dnumber}')



