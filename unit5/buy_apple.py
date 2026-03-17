import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from layer_naive import MulLayer

apple = 100 
apple_nu2 = 2 
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward 
apple_price = mul_apple_layer.forward(apple,apple_nu2)
price = mul_tax_layer.forward(apple_price,tax)

print(price)


# backward
dprice = 1 
dapple_price,dtax = mul_tax_layer.backward(dprice)
dapple,dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple,dapple_num,dtax) # 

