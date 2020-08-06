import torch
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
Matrix addition using '+' or using add
"""
a = torch.randn(3, 3)
b = torch.randn(3, 3)
print(a + b)
print(torch.add(a, b))

"""
In place operation - overwriting the existing
"""
a = torch.ones(3, 3)
print(a)
print(a.add_(a))

"""
Tensor mean & standard deviation
"""
a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(a.size())
# Check the datatype and convert into float
print(a.type())
b = torch.tensor(a, dtype=float)
print(b.mean())

a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) # For complex tensor
a = torch.tensor(a, dtype=float)
print(a.mean(dim=1))  # default is dim=0, dim=1 to take mean for both rows

print(a.std(dim=1)) # standard deviation for each row

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

For gpu computing the tensor need to be in the cpu
"""

a = torch.rand(2, 2)
print(a)
if torch.cuda.is_available():
    print("Yes cuda available")
    tensor_gpu = a.cuda()
    print(tensor_gpu.device)  # To check where the tensor is stored cpu
    tensor_cpu = tensor_gpu.cpu()
    print(tensor_cpu.device)
else:
    print("Cuda not available")

"""
Variable and gradient calculation
"""
x = torch.ones(2, requires_grad = True)

# Consider a sample equation 5(x+1)^2
y = 5*(x+1)**2

# Gradient is single value, so mean of all values takes in the tensor
mean_value = y.mean()
mean_value.backward()
print(x.grad)
