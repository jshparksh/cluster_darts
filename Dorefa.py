import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd.function import InplaceFunction

"""
    *** Custom Layer
    See https://github.com/jcjohnson/pytorch-examples#pytorch-custom-nn-modules

    *** Binary Backward Function
    See https://github.com/Akashmathwani/Binarized-Neural-networks-using-pytorch/blob/master/MNIST%20using%20Binarized%20weights/functions.py

    *** Github Quantized Neural Network Pytorch
    See https://github.com/eladhoffer/quantized.pytorch/blob/master/models/modules/quantize.py

    *** 2020.03.12 Update Reference (최종 이걸보고 구현)
    https://pytorch.org/docs/master/notes/extending.html
    
    *** 이걸 이렇게 하지 않아도 register hook으로 구현이 가능하다. children module에 register hook을 적용하여 만들 수 있다!
    https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
    
"""

"""
See https://github.com/jcjohnson/pytorch-examples#pytorch-custom-nn-modules
We can implement our own custom autograd Functions by subclassing
torch.autograd.Function and implementing the forward and backward passes
which operate on Tensors.
"""

"""
    Dorefa net 구현 : https://github.com/zzzxxxttt/pytorch_DoReFaNet/blob/master/utils/quant_dorefa.py
    Tensorflow 저자 구현: https://github.com/tensorpack/tensorpack/blob/master/examples/DoReFa-Net/dorefa.py
"""


def uniform_quantize(k):
    class qfn(InplaceFunction):
        
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)

            if k==32:
                out = input
            else:
                n = float(2.**k - 1)
                out = torch.round(input * n) / n
            return out
        
        @staticmethod
        def backward(ctx, grad_output):
            """
            input, = ctx.saved_tensors
            
            if k==32:
                # STE (already clip input to [-1, 1] so just copy the gradient)
                grad_input = grad_output.clone()
            else:
            """ 
            # STE
            grad_input = grad_output.clone()
            return grad_input
    return qfn.apply
        

class dorefa_weight(nn.Module):
    def __init__(self, num_bits):
        super(dorefa_weight, self).__init__()
        self.num_bits = num_bits
        self.uniform_q = uniform_quantize(num_bits)
        
    def forward(self, weight):
        if self.num_bits == 32:
            qw = weight
            
        # Dorefa use constant scalar to all filters when binary weight
        elif self.num_bits == 1:
            E = torch. mean(torch.abs(weight)).detach()
            qw = self.uniform_q(weight / E) * E
            
        else:         
            # limit weight to [-1, 1]
            w = torch.tanh(weight)
            max_w = torch.max(w).detach()
            w = w / 2 / max_w + 0.5
            qw = 2 * self.uniform_q(w) - 1.
        return qw
    
class dorefa_input(nn.Module):
    def __init__(self, num_bits):
        super(dorefa_input, self).__init__()
        self.num_bits = num_bits
        self.uniform_q = uniform_quantize(num_bits)
        
    def forward(self, activation):
        if self.num_bits == 32:
            qa = activation
        else:
            qa = self.uniform_q(activation.clamp_(0., 1.))
        return qa
        
class DorConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False, 
                 num_bits=4, num_bits_weight=4, num_bits_grad=4):
        super(DorConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight
        self.num_bits_grad = num_bits_grad
        
        self.qinput = dorefa_input(num_bits)
        self.qweight = dorefa_weight(num_bits_weight)
        self.qbias = dorefa_weight(num_bits_weight)
        
    def forward(self, input):
        qinput = self.qinput(input)
        qweight = self.qweight(self.weight)
        
        if self.bias is not None:
            qbias = self.qbias(self.bias)
        else:
            qbias = None
        
        return F.conv2d(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)

class DorLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False,
                 num_bits=4, num_bits_weight=4, num_bits_grad=4):
        super(DorLinear, self).__init__(in_features, out_features, bias)
        
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight
        self.num_bits_grad = num_bits_grad
        
        self.qinput = dorefa_input(num_bits)
        self.qweight = dorefa_weight(num_bits_weight)
        self.qbias = dorefa_weight(num_bits_weight)
        
    def forward(self, input):
        qinput = self.qinput(input)
        qweight = self.qweight(self.weight)
        
        if self.bias is not None:
            qbias = self.qbias(self.bias)
        else:
            qbias = None
            
        return F.linear(qinput, qweight, qbias)