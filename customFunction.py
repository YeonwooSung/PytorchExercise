import torch
import torch.nn as nn
from torch.autograd import Variable, Function


class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_variables
        return grad_output * result



if __name__ == '__main__':
    layer = Exp().apply
    print(layer)

    a = Variable(torch.Tensor([1,2]),requires_grad=True)
    output = layer(a)
    print(output)
    result = torch.sum(output)
    print(result)

    print(a.grad)
    result.backward()
    print(a.grad)
