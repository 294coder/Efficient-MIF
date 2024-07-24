import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

import sys
sys.path.append('./')

from mamba_ssm.modules.mamba_simple import Mamba
from model.base_model import BaseModel, register_model


#----------------- Refine Module--------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import math
from torch.nn import init
import os
import torchvision.transforms.functional as tf


class DenseModule(nn.Module):
    def __init__(self, channel):
        super(DenseModule, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(channel * 4, channel, 1, 1, 0)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        x3 = self.act(self.conv3(x2))
        x_final = self.conv4(torch.cat([x, x1, x2, x3], 1))

        return x_final

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.process(x)
        y = self.avg_pool(res)
        z = self.conv_du(y)
        return z *res + x
class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x
        # return z*x


class Refine(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            # CALayer(n_feat,4),
            # CALayer(n_feat,4),
            ChannelAttention(n_feat, 4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out

class FourierShift(nn.Module):
    def __init__(self, nc, shiftPixel=1):
        super(FourierShift, self).__init__()
        self.processReal = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1)
        )
        self.processImag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1)
        )
        self.output = nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1)
        self.shiftPixel = shiftPixel

    def shift(self, x_real, x_imag):
        x_realUp, x_realDown, x_realLeft, x_realRight = torch.chunk(x_real, 4, dim=1)
        x_imagUp, x_imagDown, x_imagLeft, x_imagRight = torch.chunk(x_imag, 4, dim=1)

        x_realUp = torch.roll(x_realUp, shifts=(-self.shiftPixel), dims=2)
        x_realDown = torch.roll(x_realDown, shifts=(self.shiftPixel), dims=2)
        x_realLeft = torch.roll(x_realLeft, shifts=(-self.shiftPixel), dims=3)
        x_realRight = torch.roll(x_realRight, shifts=(self.shiftPixel), dims=3)

        x_imagUp = torch.roll(x_imagUp, shifts=(-self.shiftPixel), dims=2)
        x_imagDown = torch.roll(x_imagDown, shifts=(self.shiftPixel), dims=2)
        x_imagLeft = torch.roll(x_imagLeft, shifts=(-self.shiftPixel), dims=3)
        x_imagRight = torch.roll(x_imagRight, shifts=(self.shiftPixel), dims=3)

        return torch.cat([x_realUp, x_realDown, x_realLeft, x_realRight], dim=1), torch.cat([x_imagUp, x_imagDown, x_imagLeft, x_imagRight], dim=1)

    def forward(self, x):
        x_residual = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x_real = x_freq.real
        x_imag = x_freq.imag

        x_real, x_imag = self.shift(x_real=x_real, x_imag=x_imag)

        x_processedReal = self.processReal(x_real)
        x_processedImag = self.processImag(x_imag)

        x_out = torch.complex(x_processedReal, x_processedImag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x_out = self.output(x_out)

        return x_out + x_residual

class RubikCube_multiply(nn.Module):
    def __init__(self, nc, out, shiftPixel=1, gc=1):
        super(RubikCube_multiply, self).__init__()

        self.processC1 = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.processC2 = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.processC3 = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.processC4 = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.processOutput = nn.Sequential(
            nn.Conv2d(nc, out, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.shiftPixel = shiftPixel
        self.gc = gc
        self.split_indexes = (gc, gc, gc, gc, nc - gc * 4)

    def shift_feat(self, x, shiftPixel, g):
        B, C, H, W = x.shape
        out = torch.zeros_like(x)

        out[:, g * 0:g * 1, :, :-shiftPixel] = x[:, g * 0:g * 1, :, shiftPixel:]  # shift left
        out[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]  # shift right
        out[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]  # shift up
        out[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]  # shift down

        out[:, g * 4:, :, :] = x[:, g * 4:, :, :]  # no shift
        return out

    def forward(self, x):
        residual = x
        x_shifted = self.shift_feat(x, self.shiftPixel, self.gc)
        c1, c2, c3, c4, x2 = torch.split(x_shifted, self.split_indexes, dim=1)

        c1_processed = self.processC1(c1)
        c2_processed = self.processC2(c1_processed * c2)
        c3_processed = self.processC3(c2_processed * c3)
        c4_processed = self.processC4(c3_processed * c4)

        out = torch.cat([c1_processed, c2_processed, c3_processed, c4_processed, x2], dim=1)

        # print(self.processOutput(out).size())
        return self.processOutput(out) + residual

# class RubikCube_multiply(nn.Module):
#     def __init__(self, nc, out, shiftPixel=1):
#         super(RubikCube_multiply, self).__init__()
#
#         self.processC1 = nn.Sequential(
#             nn.Conv2d(nc // 4, nc // 4, kernel_size=1, padding=0, stride=1),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#
#         self.processC2 = nn.Sequential(
#             nn.Conv2d(nc // 4, nc // 4, kernel_size=1, padding=0, stride=1),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#
#         self.processC3 = nn.Sequential(
#             nn.Conv2d(nc // 4, nc // 4, kernel_size=1, padding=0, stride=1),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#
#         self.processC4 = nn.Sequential(
#             nn.Conv2d(nc // 4, nc // 4, kernel_size=1, padding=0, stride=1),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#
#         self.processOutput = nn.Sequential(
#             nn.Conv2d(nc, out, kernel_size=1, padding=0, stride=1)
#         )
#         self.shiftPixel = shiftPixel
#
#
#     def shift(self, x):
#         x_Up, x_Down, x_Left, x_Right = torch.chunk(x, 4, dim=1)
#
#         x_Up = torch.roll(x_Up, shifts=(-self.shiftPixel), dims=2)
#         x_Down = torch.roll(x_Down, shifts=(self.shiftPixel), dims=2)
#         x_Left = torch.roll(x_Left, shifts=(-self.shiftPixel), dims=3)
#         x_Right = torch.roll(x_Right, shifts=(self.shiftPixel), dims=3)
#
#         return torch.cat([x_Up, x_Down, x_Left, x_Right], dim=1)
#
#     def forward(self, x):
#
#         residual = x
#
#         c1, c2, c3, c4 = torch.chunk(x, 4, dim=1)
#
#         c1_processed = self.processC1(c1)
#
#         c2_shifted = torch.roll(c2, shifts=(-self.shiftPixel), dims=2)
#         c2_processed = self.processC2(c1_processed * c2_shifted)
#
#         c3_shifted = torch.roll(c3, shifts=(self.shiftPixel), dims=2)
#         c3_processed = self.processC3(c2_processed * c3_shifted)
#
#         c4_shifted = torch.roll(c4, shifts=(-self.shiftPixel), dims=3)
#         c4_processed = self.processC4(c3_processed * c4_shifted)
#
#         out = torch.cat([c1_processed, c2_processed, c3_processed, c4_processed], dim=1)
#
#         out = out + residual
#
#         return self.processOutput(out)

class RefineRubik(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(RefineRubik, self).__init__()

        # self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.conv_in = RubikCube_multiply(n_feat,n_feat)
        self.process = nn.Sequential(
            # CALayer(n_feat,4),
            # CALayer(n_feat,4),
            ChannelAttention(n_feat, 4))
        # self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Sequential(RubikCube_multiply(n_feat,n_feat),nn.Conv2d(n_feat,out_channel,1,1,0))
    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)
        return out
class RefineShift(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(RefineShift, self).__init__()

        # self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.conv_in = FourierShift(n_feat)
        self.process = nn.Sequential(
            # CALayer(n_feat,4),
            # CALayer(n_feat,4),
            ChannelAttention(n_feat, 4))
        # self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Sequential(FourierShift(n_feat),nn.Conv2d(n_feat,out_channel,1,1,0))
    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out

class Refine1(nn.Module):

    def __init__(self, in_channels, panchannels, n_feat):
        super(Refine1, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            # CALayer(n_feat,4),
            # CALayer(n_feat,4),
            CALayer(n_feat, 4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=in_channels - panchannels, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out
    
# ---------------------------------------------------------------------------------------------------------------------


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ms, pan):
        b, c, h, w = ms.shape

        kv = self.kv_dwconv(self.kv(pan))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(ms))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm_cro1= LayerNorm(dim, LayerNorm_type)
        self.norm_cro2 = LayerNorm(dim, LayerNorm_type)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.cro = CrossAttention(dim,num_heads,bias)
        self.proj = nn.Conv2d(dim,dim,1,1,0)
    def forward(self, ms,pan):
        ms = ms+self.cro(self.norm_cro1(ms),self.norm_cro2(pan))
        ms = ms + self.ffn(self.norm2(ms))
        return ms


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# ---------------------------------------------------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)
class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x
class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim,bimamba_type=None)
        self.norm = LayerNorm(dim,'with_bias')
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
    def forward(self,ipt):
        x,residual = ipt
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)
class TokenSwapMamba(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.msencoder = Mamba(dim,bimamba_type=None)
        self.panencoder = Mamba(dim,bimamba_type=None)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
    def forward(self, ms,pan
                ,ms_residual,pan_residual):
        # ms (B,N,C)
        #pan (B,N,C)
        ms_residual = ms+ms_residual
        pan_residual = pan+pan_residual
        ms = self.norm1(ms_residual)
        pan = self.norm2(pan_residual)
        B,N,C = ms.shape
        ms_first_half = ms[:, :, :C//2]
        pan_first_half = pan[:, :, :C//2]
        ms_swap= torch.cat([pan_first_half,ms[:,:,C//2:]],dim=2)
        pan_swap= torch.cat([ms_first_half,pan[:,:,C//2:]],dim=2)
        ms_swap = self.msencoder(ms_swap)
        pan_swap = self.panencoder(pan_swap)
        return ms_swap,pan_swap,ms_residual,pan_residual
class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim,bimamba_type="v3")
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    def forward(self,ms,ms_resi,pan):
        ms_resi = ms+ms_resi
        # modified
        ms = self.norm1(ms_resi)
        pan = self.norm2(pan)
        global_f = self.cross_mamba(self.norm1(ms),extra_emb=self.norm2(pan))
        B,HW,C = global_f.shape
        H = W = int(math.sqrt(HW))
        ms = global_f.transpose(1, 2).view(B, C, H, W)
        ms =  (self.dwconv(ms)+ms).flatten(2).transpose(1, 2)
        return ms,ms_resi
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi
    
@register_model('panmamba_zhouman')
class Net(BaseModel):
    def __init__(self,num_channels=None,base_filter=None,args=None):
        super(Net, self).__init__()
        base_filter=32
        self.base_filter = base_filter
        self.stride=1
        self.patch_size=1
        self.pan_encoder = nn.Sequential(nn.Conv2d(1,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.ms_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.embed_dim = base_filter*self.stride*self.patch_size
        self.shallow_fusion1 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.shallow_fusion2 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.ms_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.pan_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.deep_fusion1= CrossMamba(self.embed_dim)
        self.deep_fusion2 = CrossMamba(self.embed_dim)
        self.deep_fusion3 = CrossMamba(self.embed_dim)
        self.deep_fusion4 = CrossMamba(self.embed_dim)
        self.deep_fusion5 = CrossMamba(self.embed_dim)

        self.pan_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])
        self.ms_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])
        self.swap_mamba1 = TokenSwapMamba(self.embed_dim)
        self.swap_mamba2 = TokenSwapMamba(self.embed_dim)
        self.patchunembe = PatchUnEmbed(base_filter)
        self.output = Refine(base_filter,num_channels)
    def _forward_implem(self,ms,pan):

        # modifed
        # ms_bic = F.interpolate(ms,scale_factor=4)
        ms_f = self.ms_encoder(ms)
        # ms_f = ms_bic
        # pan_f = pan
        b,c,h,w = ms_f.shape
        pan_f = self.pan_encoder(pan)
        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)
        residual_ms_f = 0
        residual_pan_f = 0
        ms_f,residual_ms_f = self.ms_feature_extraction([ms_f,residual_ms_f])
        pan_f,residual_pan_f = self.pan_feature_extraction([pan_f,residual_pan_f])
        ms_f,pan_f,residual_ms_f,residual_pan_f = self.swap_mamba1(ms_f,pan_f,residual_ms_f,residual_pan_f)
        ms_f,pan_f,residual_ms_f,residual_pan_f = self.swap_mamba2(ms_f,pan_f,residual_ms_f,residual_pan_f)
        ms_f = self.patchunembe(ms_f,(h,w))
        pan_f = self.patchunembe(pan_f,(h,w))
        ms_f = self.shallow_fusion1(torch.concat([ms_f,pan_f],dim=1))+ms_f
        pan_f = self.shallow_fusion2(torch.concat([pan_f,ms_f],dim=1))+pan_f
        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)
        residual_ms_f = 0
        ms_f,residual_ms_f = self.deep_fusion1(ms_f,residual_ms_f,pan_f)
        ms_f,residual_ms_f = self.deep_fusion2(ms_f,residual_ms_f,pan_f)
        ms_f,residual_ms_f = self.deep_fusion3(ms_f,residual_ms_f,pan_f)
        ms_f,residual_ms_f = self.deep_fusion4(ms_f,residual_ms_f,pan_f)
        ms_f,residual_ms_f = self.deep_fusion5(ms_f,residual_ms_f,pan_f)
        ms_f = self.patchunembe(ms_f,(h,w))
        hrms = self.output(ms_f)+ms
        return hrms
    
    def train_step(self, ms, lms, pan, gt, criterion):
        pred = self._forward_implem(lms, pan)
        loss = criterion(pred, gt)
        return pred.clip(0, 1), loss

    def val_step(self, ms, lms, pan, patch_merge=False):
        assert not patch_merge, 'patch merge is not supported for validation'
        
        if  patch_merge:
            pred = self._patch_merge_model.forward_chop(ms, lms, pan)[0]
        else:
            pred = self._forward_implem(lms, pan)

        return pred.clip(0, 1)

    def patch_merge_step(self, ms, lms, pan):
        return self._forward_implem(lms, pan)
    
    
if __name__ == '__main__':
    torch.cuda.set_device(1)
    
    net = Net(8, 32).cuda()
    
    lms = torch.randn(1, 8, 64, 64).cuda()
    pan = torch.randn(1, 1, 64, 64).cuda()
    gt = torch.randn(1, 8, 64, 64).cuda()
    
    print(net(lms, pan).shape)
    sr = net._forward_implem(lms, pan)
    loss = F.mse_loss(sr, gt)
    loss.backward()
    
    # print(loss)
    
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    
    net.forward = net._forward_implem
    print(flop_count_table(FlopCountAnalysis(net, (lms, pan))))

