from __future__ import print_function
import torch

# Construct a 5 X 3 uninitialised matrix
x = torch.empty(5, 3)
print(x)

# Construct a randomly initialized matrix
x = torch.rand(5, 3)
print(x)

# Construct a matrix filled zeros and of dtype long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# Construct a tensor directly from data
x = torch.tensor([5.5, 3])
print(x)

# Create a tensor based on an existing tensor
x = x.new_ones(5, 3, dtype=torch.double)
print(x)


x = torch.randn_like(x, dtype=torch.float)  # override dtype!
print(x)


y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

# adds y to x
y.add_(x)
print(y)


# Resizing: If you want to resize/reshape tensor, you can use torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# By using a -1, we are being lazy in doing the computation ourselves and rather delegate the task to 
# PyTorch to do calculation of that value for the shape when it creates the new view.
print(x.size(), y.size(), z.size())


# If you have a one element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
print(x)
print(x.item())
