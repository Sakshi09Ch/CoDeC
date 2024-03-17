import torch
import torch.nn as nn
import math


class EvoNormBatch2d(nn.Module):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-5, drop_block=None):
        super(EvoNormBatch2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum  = momentum
        self.eps       = eps
        param_shape    = (1, num_features, 1, 1)
        self.weight    = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias      = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        x_type = x.dtype
        if self.training:
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            n = x.numel() / x.shape[1]
            self.running_var.copy_(
                var.detach() * self.momentum * (n / (n - 1)) + self.running_var * (1 - self.momentum))
        else:
            var = self.running_var

        if self.apply_act:
            d = x + (x.var(dim=(2, 3), unbiased=False, keepdim=True) + self.eps).sqrt().to(dtype=x_type)
            d = d.max((var + self.eps).sqrt().to(dtype=x_type))
            x = x / d
        return x * self.weight + self.bias


class EvoNormSample2d(nn.Module):
    def __init__(self, num_features, apply_act=True, groups=8, eps=1e-5, drop_block=None, num_bits=None, num_bits_grad=None):
        super(EvoNormSample2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.groups    = groups
        self.eps       = eps
        param_shape    = (1, num_features, 1, 1)
        self.weight    = nn.Parameter(torch.ones(param_shape), requires_grad=True)  # gamma
        self.bias      = nn.Parameter(torch.zeros(param_shape), requires_grad=True)  # beta
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        B, C, H, W = x.shape
        assert C % self.groups == 0
        if self.apply_act:
            n = x * x.sigmoid() # n = x*sigmoid(x*v)
            x = x.reshape(B, self.groups, -1)
            x = n.reshape(B, self.groups, -1) / (x.var(dim=-1, unbiased=False, keepdim=True) + self.eps).sqrt()  # x = n/var(x) groupwise instance variance
            x = x.reshape(B, C, H, W)
        return x * self.weight + self.bias # gamma*x+beta



