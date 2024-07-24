# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence
import warnings
import os

import math
import logging
import torch
import triton
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp

from torch.utils.cpp_extension import load

from model.module.csm_triton import triton_cross_scan, triton_cross_merge

# from mmcv.runner import load_checkpoint
# from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed
# from mmseg.models.builder import BACKBONES
# from mmseg.utils import get_root_logger

# print('loading vrwkv module')
T_MAX = 8192 # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice
cwd = os.path.dirname(__file__)
wkv_cuda = load(name="wkv", sources=[f"{cwd}/rwkv_cuda/wkv_op.cpp", f"{cwd}/rwkv_cuda/wkv_cuda.cu"], build_directory=cwd,
                verbose=False, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class CrossScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = min(triton.next_power_of_2(C), 2), min(triton.next_power_of_2(H), 32), min(triton.next_power_of_2(W), 32)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        x = x.contiguous()
        y = x.new_empty((B, 4, C, H, W))
        triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return y.view(B, 4, C, -1)
    
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = y.contiguous().view(B, 4, C, H, W)
        x = y.new_empty((B, C, H, W))
        triton_cross_merge[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return x


class CrossMergeTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor):
        B, K, C, H, W = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = min(triton.next_power_of_2(C), 2), min(triton.next_power_of_2(H), 32), min(triton.next_power_of_2(W), 32)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        y = y.contiguous().view(B, 4, C, H, W)
        x = y.new_empty((B, C, H, W))
        triton_cross_merge[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return x.view(B, C, -1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # out: (b, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        x = x.contiguous()
        y = x.new_empty((B, 4, C, H, W))
        triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return y



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()

        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        # import pdb
        # pdb.set_trace()
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (gy.dtype == torch.half)
        wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad(): # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1)) # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                
                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                
                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, patch_resolution)
        rwkv = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        rwkv = self.output(rwkv)
        return rwkv


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution):
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False,
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, shift_pixel, init_mode,
                                   key_norm=key_norm)
        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, shift_pixel, hidden_rate,
                                   init_mode, key_norm=key_norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, patch_resolution):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

# @BACKBONES.register_module()
class VRWKV(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 embed_dims=256,
                 depth=12,
                 drop_path_rate=0.,
                 channel_gamma=1/4,
                 shift_pixel=1,
                 shift_mode='q_shift',
                 init_mode='fancy',
                 post_norm=False,
                 key_norm=False,
                 init_values=None,
                 hidden_rate=4,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = depth
        self.drop_path_rate = drop_path_rate

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=True)
        
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dims))
        
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(Block(
                n_embd=embed_dims,
                n_layer=depth,
                layer_id=i,
                channel_gamma=channel_gamma,
                shift_pixel=shift_pixel,
                shift_mode=shift_mode,
                hidden_rate=hidden_rate,
                drop_path=dpr[i],
                init_mode=init_mode,
                post_norm=post_norm,
                key_norm=key_norm,
                init_values=init_values,
                with_cp=with_cp
            ))

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(self.embed_dims)


    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # x = x + resize_pos_embed(
        #     self.pos_embed,
        #     self.patch_resolution,
        #     patch_resolution,
        #     mode=self.interpolate_mode,
        #     num_extra_tokens=self.num_extra_tokens)
        
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, patch_resolution)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)

                out = patch_token
                outs.append(out)
        return tuple(outs)

if __name__ == '__main__':
    device = 'cuda:0'
    net = Block(32, 2, 1).to(device)
    x = torch.randn(1, 64*64, 32).to(device)
    
    # print(net(x, (64, 64)).shape)
    y = net(x, (64, 64))
    sr = torch.randn(1, 64*64, 32).to(device)
    loss = F.mse_loss(y, sr)
    print(loss)
    loss.backward()
