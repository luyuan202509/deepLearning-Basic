import torch 

tensor = torch.randn(3,2)
print(tensor)
"""
print(torch.sum(tensor)) # 求和
print(torch.mean(tensor)) # 求平均
print(torch.max(tensor)) # 求最大值
print(torch.min(tensor)) # 求最小值
print(torch.std(tensor)) # 求标准差
print(torch.var(tensor)) # 求方差
print(torch.median(tensor)) # 求中位数
print(torch.mode(tensor)) # 求众数
print(torch.quantile(tensor, 0.75)) # 求 75% 分位数
print(torch.median(tensor)) # 求中位数
"""

print(tensor.argmax()) # 求最大值的索引
print(tensor.argmin()) # 求最小值的索引

# tensor 展平
print(tensor.view(6)) # 展平

# 求平方
print(tensor.pow(2)) # 求平方