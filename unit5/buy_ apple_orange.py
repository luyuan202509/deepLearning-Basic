
from this import d
from  layer_naive import MulLayer,AddLayer

apple = 100
apple_num = 2
orange = 150
orange_num =3
tax = 1.1

#layers 
mul_apple_layer = MulLayer()
num_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward

apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = num_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

print('*****************向前传播，每个节点输入数据***************')
print(f'苹果的价格：{apple_price}')
print('橘子的价格：', orange_price)
print('所有商品的价格：', all_price)
print('总价格：', price)


print('*****************反向传播，每个节点输出是梯度（导数）***************')
#backward
dprice =1
dall_price, dtax = mul_tax_layer.backward(dprice)
deapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backward(deapple_price)
dorange, dorange_num = num_orange_layer.backward(dorange_price)

print(f'价格的梯度：{dall_price}')
print(f'消费税的梯度：{dtax}')
print(f'苹果价格的梯度：{deapple_price}')
print(f'橘子价格的梯度：{dorange_price}')

print(f'单个苹果价格的梯度：{dapple}')
print(f'单个橘子价格的梯度：{dorange}')

print(f'苹果数量的梯度：{dapple_num}')
print(f'橘子数量的梯度：{dorange_num}')

