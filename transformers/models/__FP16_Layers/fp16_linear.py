import jittor as jt
import numpy as np
from jittor import Module, init, Function
import torch

### model define
class FP16_Linear_Function(Function): 
    # X: FP16 / BF16
    # W: FP32
    # intermediate: FP16 / BF16
    # out: FP32
    def __init__(self):
        super(FP16_Linear_Function, self).__init__()
        self.saved = None 
    
    def execute(self, x, weight, bias, use_fp16):
        # assert x.dtype == jt.float16 if use_fp16 else jt.bfloat16
        if x.dtype != jt.float16 if use_fp16 else jt.bfloat16:
            x = x.float16() if use_fp16 else x.bfloat16()       # turn into 16bit
        weight = weight.float16() if use_fp16 else weight.bfloat16()  # turn into 16bit
        bias   = bias.float16()   if use_fp16 else bias.bfloat16()    # turn into 16bit
        
        self.saved = x, weight, bias, use_fp16
        return jt.nn.linear(x, weight, bias)
    
    def grad(self, grad_output):
        x, weight, bias, use_fp16 = self.saved
        if grad_output.dtype != jt.float16 if use_fp16 else jt.bfloat16:
            grad_output = grad_output.float16() if use_fp16 else grad_output.bfloat16() # turn into 16bit
        
        x_flatten = x.reshape(-1, x.shape[-1])
        grad_output_flatten = grad_output.reshape(-1, grad_output.shape[-1])
        
        grad_input  = grad_output_flatten @ weight
        grad_weight = grad_output_flatten.t() @ x_flatten
        
        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None
        
        grad_input_transform = grad_input.reshape(x.size())
        return grad_input_transform, grad_weight.float32(), grad_bias


class FP16_Matmul_Function(Function): 
    # in: FP32
    # intermediate: FP16 / BF16
    # out: FP32
    def __init__(self):
        super(FP16_Matmul_Function, self).__init__()
        self.saved = None 
    
    def execute(self, x, y, use_fp16):
        if x.dtype != jt.float16 if use_fp16 else jt.bfloat16:
            x = x.float16() if use_fp16 else x.bfloat16()  # turn into 16bit
        if y.dtype != jt.float16 if use_fp16 else jt.bfloat16:
            y = y.float16() if use_fp16 else y.bfloat16()  # turn into 16bit
        
        self.saved = x, y, use_fp16
        return x @ y
    
    def grad(self, grad_output):
        x, y, use_fp16 = self.saved
        grad_output = grad_output.float16() if use_fp16 else grad_output.bfloat16() # turn into 16bit
        
        grad_x = jt.matmul(grad_output, y.t())  # 对 x 的梯度
        grad_y = jt.matmul(x.t(), grad_output)
        
        return grad_x, grad_y


class FP16_Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, use_fp16=True): # use_fp16=False : BF16
        super(FP16_Linear, self).__init__(in_features, out_features, bias)
        self.use_fp16 = use_fp16
        
    def execute(self, input):
        return FP16_Linear_Function.apply(input, self.weight, self.bias, self.use_fp16)

class FP16_Matmul(torch.nn.Module):
    def __init__(self, use_fp16=True): # use_fp16=False : BF16
        super(FP16_Matmul, self).__init__()
        self.use_fp16 = use_fp16
        
    def forward(self, x, y):
        return FP16_Matmul_Function.apply(x, y, self.use_fp16)
