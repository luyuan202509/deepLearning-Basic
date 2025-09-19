import os,sys 
current_dir = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在目录
review_dir = os.path.dirname(current_dir) # review目录
project_root = os.path.dirname(review_dir) # 项目根目录
sys.path.append(project_root)
from review.unit05.buy_apple import apple_price, dapple, dapple_price
from review.unit05.layer_naive import MulLayer,AddLayer

 
 
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

 # layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price =  mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

print(f"2苹果价格: {apple_price}")
print(f"3橘子价格: {orange_price}")
print(f"2苹果3橘子价格: {all_price}")
print(f"2苹果3橘子价格(含税): {price}")

# backward
dprice = 1
dall_price, d_tax = mul_apple_layer.backward(dprice)
dapple_price_all, d_orange_price_all= mul_apple_layer.backward(dprice)
dapple, d_apple_num = mul_apple_layer.backward(dapple_price_all)
deorange, d_orange_num = mul_orange_layer.backward(d_orange_price_all)

print(f"含税总价格梯度: {dall_price}")
print(f"税额梯度: {d_tax}")
print(f"苹果总价格梯度: {dapple_price_all}")
print(f"橘子总价格梯度: {d_orange_price_all}")
print(f"苹果梯度: {dapple}")
print(f"苹果数量梯度: {d_apple_num}")
print(f"橘子梯度: {dapple}")
print(f"橘子数量梯度: {d_orange_num}")



    




