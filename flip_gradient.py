from torch import nn
import torch


class GradReverse(torch.autograd.Function):
    rate = 0.0

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return args[0].view_as(args[0])

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0].neg()*GradReverse.rate
        return grad_output, None


class GRL(nn.Module):
    @staticmethod
    def forward(inp):
        return GradReverse.apply(inp)
