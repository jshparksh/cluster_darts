""" Operations """
import torch
import torch.nn as nn
import genotypes as gt
import os
from feature_map import save_features
from Dorefa import *
from PACT import *

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'conv_3x3_16' : lambda C, stride, affine: ConvQ(C, C, kernel_size=3, stride=stride, padding=1, 
                        num_bits=16, num_bits_weight=16, num_bits_grad=16, affine=affine),
    'conv_3x3_8' : lambda C, stride, affine: ConvQ(C, C, kernel_size=3, stride=stride, padding=1, 
                        num_bits=8, num_bits_weight=8, num_bits_grad=8, affine=affine),
    'conv_3x3_4' : lambda C, stride, affine: ConvQ(C, C, kernel_size=3, stride=stride, padding=1, 
                        num_bits=4, num_bits_weight=4, num_bits_grad=4, affine=affine),
    'sep_conv_3x3_16' : lambda C, stride, affine: SepConvQ(C, C, kernel_size=3, stride=stride, padding=1, 
                        num_bits=16, num_bits_weight=16, num_bits_grad=16, affine=affine),
    'sep_conv_3x3_8' : lambda C, stride, affine: SepConvQ(C, C, kernel_size=3, stride=stride, padding=1, 
                        num_bits=8, num_bits_weight=8, num_bits_grad=8, affine=affine),
    'sep_conv_3x3_4' : lambda C, stride, affine: SepConvQ(C, C, kernel_size=3, stride=stride, padding=1, 
                        num_bits=4, num_bits_weight=4, num_bits_grad=4, affine=affine),
    'dil_conv_3x3_16' : lambda C, stride, affine: DilConvQ(C, C, 3, stride, 2, 2, 
                        num_bits=16, num_bits_weight=16, num_bits_grad=16, affine=affine),
    'dil_conv_3x3_8' : lambda C, stride, affine: DilConvQ(C, C, 3, stride, 2, 2, 
                        num_bits=8, num_bits_weight=8, num_bits_grad=8, affine=affine),
    'dil_conv_3x3_4' : lambda C, stride, affine: DilConvQ(C, C, 3, stride, 2, 2, 
                        num_bits=4, num_bits_weight=4, num_bits_grad=4, affine=affine)
}
"""
    'sep_conv_5x5_8' : lambda C, stride, affine: SepConvQ(C, C, kernel_size=5, stride=stride, padding=2, 
                        num_bits=8, num_bits_weight=8, num_bits_grad=8, affine=affine),
    'sep_conv_5x5_4' : lambda C, stride, affine: SepConvQ(C, C, kernel_size=5, stride=stride, padding=2, 
                        num_bits=4, num_bits_weight=4, num_bits_grad=4, affine=affine),
    'dil_conv_5x5_8' : lambda C, stride, affine: DilConvQ(C, C, 5, stride, 4, 2, 
                        num_bits=8, num_bits_weight=8, num_bits_grad=8, affine=affine),
    'dil_conv_5x5_4' : lambda C, stride, affine: DilConvQ(C, C, 5, stride, 4, 2, 
                        num_bits=4, num_bits_weight=4, num_bits_grad=4, affine=affine),
"""

def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)
    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)

class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)
    
class ConvQ(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, num_bits, num_bits_weight, num_bits_grad, affine=True):
        super(ConvQ, self).__init__()
        self.net = nn.Sequential(
            PACT_with_quantize(num_bits=num_bits_weight),#nn.ReLU(inplace=False),
            DorConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, 
                        num_bits=num_bits, num_bits_weight= num_bits_weight, num_bits_grad = num_bits_grad),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.net(x)
    
class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)
    
class DilConvQ(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, num_bits, num_bits_weight, num_bits_grad, affine=True):
        super(DilConvQ, self).__init__()
        self.net = nn.Sequential(
            PACT_with_quantize(num_bits=num_bits_weight),#nn.ReLU(inplace=False),
            DorConv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False, 
                        num_bits=num_bits, num_bits_weight= num_bits_weight, num_bits_grad = num_bits_grad),
            DorConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False, 
                        num_bits=num_bits, num_bits_weight= num_bits_weight, num_bits_grad = num_bits_grad),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.net(x)

class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)

class SepConvQ(nn.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, num_bits, num_bits_weight, num_bits_grad, affine=True):
        super(SepConvQ, self).__init__()
        self.net = nn.Sequential(
            PACT_with_quantize(num_bits=num_bits_weight),#nn.ReLU(inplace=False),
            DorConv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False,
                        num_bits=num_bits, num_bits_weight= num_bits_weight, num_bits_grad = num_bits_grad),
            DorConv2d(C_in, C_in, kernel_size=1, padding=0, bias=False, 
                        num_bits=num_bits, num_bits_weight= num_bits_weight, num_bits_grad = num_bits_grad),
            nn.BatchNorm2d(C_in, affine=affine),
            PACT_with_quantize(num_bits=num_bits_weight),#nn.ReLU(inplace=False),
            DorConv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False,
                        num_bits=num_bits, num_bits_weight= num_bits_weight, num_bits_grad = num_bits_grad),
            DorConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False, 
                        num_bits=num_bits, num_bits_weight= num_bits_weight, num_bits_grad = num_bits_grad),
            nn.BatchNorm2d(C_out, affine=affine)
            
        )
    
    def forward(self, x):
        return self.net(x)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class MixedOp(nn.Module):
    """ Mixed operation """
    def __init__(self, C, stride): #, cell_id, node_id, edge_id): #maybe this vars will be needed to save features for visualization
        super().__init__()
        self._ops = nn.ModuleList()
        self._feature = []
        self.C = C
        self.stride = stride
        
        for primitive in gt.PRIMITIVES_FIRST:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        self._feature = [op(x) for op in self._ops]
        
        return sum(w * op(x) for w, op in zip(weights, self._ops))
    
    def feature(self):
        return self._feature

class MixedOp_Fixed(nn.Module):
    """ Mixed operation """
    def __init__(self, C, stride): #, cell_id, node_id, edge_id): #maybe this vars will be needed to save features for visualization
        super().__init__()
        self._ops = nn.ModuleList()
        self._feature = []
        self.C = C
        self.stride = stride
        
        for primitive in gt.PRIMITIVES_SECOND:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        self._feature = [op(x) for op in self._ops]
        
        return sum(w * op(x) for w, op in zip(weights, self._ops))
    
    def feature(self):
        return self._feature
