import torch
import numpy as np

"""
Arange - for generating data within a limit
view - for reshaping the arrays
"""

a = torch.arange(18).view(3, 2, 3)
print(a)
print(a.shape)

"""
Matrix multiplication using torch.matmul(a,b) or using a@b
"""
a = torch.tensor([0, 3, 5, 4, 5, 2]).view(3, 2)
b = torch.tensor([3, 4, -3, -2, 4, -2]).view(2, 3)
print(torch.matmul(a, b))
print(a @ b)

"""
Using seed to generate constant set of rands
"""
print(torch.randn(3, 3))
torch.manual_seed(0)
print(torch.randn(3, 3))

"""
Converting between numpy and torch tensors
"""
a = np.random.randn(3, 3)
print(a)
print(type(a))
torch_tensor = torch.from_numpy(a)
print(torch_tensor)
print(type(torch_tensor))
numpy_array = torch_tensor.numpy()
print(numpy_array)

"""
Moving tensors to CPU <--> GPU
"""
