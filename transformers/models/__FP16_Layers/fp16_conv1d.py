import jittor as jt
import numpy as np
from jittor import Module, init, Function

import torch
from torch import nn

class FP16_Conv1d_Function(Function): 
    # X: FP16 / BF16
    # W: FP32
    # intermediate: FP16 / BF16
    # out: FP32
    def __init__(self):
        super(FP16_Conv1d_Function, self).__init__()
        self.saved = None 
    
    def execute(self, x, weight, bias, nf, use_fp16):
        # assert x.dtype == jt.float16 if use_fp16 else jt.bfloat16
        if x.dtype != jt.float16 if use_fp16 else jt.bfloat16:
            x = x.float16() if use_fp16 else x.bfloat16()       # turn into 16bit
        weight = weight.float16() if use_fp16 else weight.bfloat16()  # turn into 16bit
        bias   = bias.float16()   if use_fp16 else bias.bfloat16()    # turn into 16bit
        
        self.saved = x, weight, bias, use_fp16
        
        size_out = x.size()[:-1] + (nf,)
        x = bias + x.view(-1, x.size(-1)) @ weight
        x = x.view(*size_out)
        return x
    
    def grad(self, grad_output):
        x, weight, bias, use_fp16 = self.saved
        if grad_output.dtype != jt.float16 if use_fp16 else jt.bfloat16:
            grad_output = grad_output.float16() if use_fp16 else grad_output.bfloat16() # turn into 16bit
        
        x_flatten = x.reshape(-1, x.shape[-1])
        grad_output_flatten = grad_output.reshape(-1, grad_output.shape[-1])
        
        grad_input  = grad_output_flatten @ weight.t()
        grad_weight = x_flatten.t() @ grad_output_flatten
        
        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None
        
        grad_input_transform = grad_input.reshape(x.size())
        return grad_input_transform, grad_weight.float32(), grad_bias

class FP16_Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx, use_fp16=True): # Use BF16 Here !!!?
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        # nn.init.normal_(w, std=0.02)
        torch.init.gauss_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))
        self.use_fp16 = use_fp16
        
    def forward(self, x):
        return FP16_Conv1d_Function.apply(x, self.weight, self.bias, self.nf, self.use_fp16)
