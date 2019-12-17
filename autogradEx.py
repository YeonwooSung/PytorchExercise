import torch
from torch.autograd import Variable


# The "Variable" class is the most important class of the autograd package.
# Basically, the Variable class covers the tensor, and it supports all operations for the tensor.
# The Variable class has members "data", "grad", and "grad_fn".
#   1) Variable.data    =>  The actual data that the tensor contains.
#   2) Variable.grad    =>  The gradient for the tensor.
#   3) Variable.grad_fn =>  An attribute that references a Function that has created the tensor.
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)


# Create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)

# tensor operation
y = x + 2
print(y)
print(y.grad_fn)


# The method requires_grad_( <boolean_value> ) changes an existing Tensor’s requires_grad flag in-place.
# The input flag defaults to False if not given.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
