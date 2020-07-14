"""
Arange - for generating data within a limit
view - for reshaping the arrays
"""
import torch

a = torch.arange(18).view(3, 2, 3)
print(a)
print(a.shape)

"""
Matrix multiplication using torch.matmul(a,b) or using a@b
"""
a = torch.rand(2, 2)
b = torch.randn(2, 2)
print(torch.matmul(a, b))
print(a @ b)

