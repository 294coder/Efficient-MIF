########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import triton
import torch.nn as nn
from torch.nn import functional as F
# import pytorch_lightning as pl
# from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
# from pytorch_lightning.strategies import DeepSpeedStrategy
# if importlib.util.find_spec('deepspeed'):
#     import deepspeed
#     from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

import sys

sys.path.append('./')

from model.module.csm_triton import (triton_cross_scan,
                                     triton_cross_merge,
                                     triton_cross_scan_same_and_trans,
                                     triton_cross_merge_trans_and_flips,
                                     triton_cross_merge_same_and_trans,
                                     triton_cross_scan_trans_and_flips)

# perfixed os env
os.environ["RWKV_MY_TESTING"] = 'x060'
os.environ["RWKV_JIT_ON"] = "0"
os.environ["RWKV_HEAD_SIZE_A"] = "32"
os.environ['RWKV_CTXLEN'] = "16384"

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
    
    
def antidiagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

def diagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

def diagonal_scatter(tensor_flat, original_shape):
    # 把斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 创建一个空的张量来存储反向散布的结果
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, H, W]，考虑到需要使用transpose将H和W调换
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_根据expanded_index将元素放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

def antidiagonal_scatter(tensor_flat, original_shape):
    # 把反斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 初始化一个与原始张量形状相同、元素全为0的张量
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, W, H]，因为操作是沿最后一个维度收集的，需要调整形状并交换维度
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_将元素根据索引放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


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

class CrossScan(torch.autograd.Function):
    # ZSJ 这里是把图像按照特定方向展平的地方，改变扫描方向可以在这里修改
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # xs = x.new_empty((B, 4, C, H * W))
        xs = x.new_empty((B, 8, C, H * W))
        # 添加横向和竖向的扫描
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
    
        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])
        
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        # 把横向和竖向的反向部分再反向回来，并和原来的横向和竖向相加
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B,C,H,W)) + antidiagonal_scatter(y_da[:, 1], (B,C,H,W))

        y_res = y_rb + y_da
        # return y.view(B, -1, H, W)
        return y_res

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

class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)

        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B,D,H,W)) + antidiagonal_scatter(y_da[:, 1], (B,D,H,W))

        y_res = y_rb + y_da
        return y_res.view(B, D, -1)
        # return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        # xs = x.new_empty((B, 4, C, L))
        xs = x.new_empty((B, 8, C, L))

        # 横向和竖向扫描
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        # xs = xs.view(B, 4, C, H, W)

        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x.view(B,C,H,W))
        xs[:, 5] = antidiagonal_gather(x.view(B,C,H,W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        # return xs
        return xs.view(B, 8, C, H, W)


# Vmamba cross scan
    

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


class CrossScanTritonSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scan_mode: int=0):
        ctx.scan_mode = scan_mode
        B, C, H, W = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = min(triton.next_power_of_2(C), 2), min(triton.next_power_of_2(H), 32), min(triton.next_power_of_2(W), 32)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        x = x.contiguous()
        y = x.new_empty((B, 2, C, H, W))
        if scan_mode == 0:
            triton_cross_scan_same_and_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif scan_mode == 1:
            triton_cross_scan_trans_and_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        else:
            raise RuntimeError('scan_mode should be 1 or 2')
        
        return y.view(B, 2, C, -1)
    
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l) 
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = y.contiguous().view(B, 2, C, H, W)
        x = y.new_empty((B, C, H, W))
        if ctx.scan_mode == 0:
            triton_cross_merge_same_and_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif ctx.scan_mode == 1:
            triton_cross_merge_trans_and_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        else:
            raise RuntimeError('scan_mode should be 1 or 2')
        
        return x, None

class CrossMergeTritonSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, scan_mode: int=0):
        ctx.scan_mode = scan_mode
        B, K, C, H, W = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = min(triton.next_power_of_2(C), 2), min(triton.next_power_of_2(H), 32), min(triton.next_power_of_2(W), 32)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        y = y.contiguous().view(B, 2, C, H, W)
        x = y.new_empty((B, C, H, W))
        if scan_mode == 0:
            triton_cross_merge_same_and_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif scan_mode == 1:
            triton_cross_merge_trans_and_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
            
        return x.view(B, C, -1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # out: (b, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        x = x.contiguous()
        y = x.new_empty((B, 2, C, H, W))
        if ctx.scan_mode == 0:
            triton_cross_scan_same_and_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif ctx.scan_mode == 1:
            triton_cross_scan_trans_and_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
            
        return y, None

@torch.jit.script
def groups_q_shift(input: torch.Tensor,
                   shift_pixel: int=1,
                   gamma: float=1/4,
                   H: int=64, 
                   W: int=64):
    assert gamma <= 1/4
    K = 4
    B, KC, N = input.size()
    C = KC // 4
    input = input.reshape(B, K, C, H, W)
    output = torch.zeros_like(input)
    output[..., 0:int(C*gamma), :, shift_pixel:W] = input[..., 0:int(C*gamma), :, 0:W-shift_pixel]
    output[..., int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[..., int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[..., int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[..., int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[..., int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[..., int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[..., int(C*gamma*4):, :, :] = input[..., int(C*gamma*4):, :, :]
    output = output.flatten(3)
    output = output.view(B, KC, H*W)
    return output

class ShiftByConv(nn.Module):
    def __init__(self, chan,):
        super().__init__()
        self.dwconv = nn.Conv2d(chan, chan, kernel_size=3, stride=1, padding=1, groups=chan)
        
    def forward(self, x, x_h, x_w):
        bs, c = x.size(0), x.size(1)
        x = x.view(bs, c, x_h, x_w)
        x = self.dwconv(x)
        x = x.view(bs, c, -1)
        
        return x

########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])

if 'x060' in os.environ["RWKV_MY_TESTING"]:
    cuda_file = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv6_cuda.cu"
    cpp_file = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv6_op.cpp"
    os.makedirs(f"{os.path.dirname(__file__)}/rwkv_cuda/build", exist_ok=True)
    
    wkv6_cuda = load(name="wkv6", sources=[cuda_file, cpp_file],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"],
                    build_directory=f"{os.path.dirname(__file__)}/rwkv_cuda/build")
        
    class WKV_6(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u):
            with torch.no_grad():
                assert r.dtype == torch.float32
                assert k.dtype == torch.float32
                assert v.dtype == torch.float32
                assert w.dtype == torch.float32
                assert u.dtype == torch.float32
                assert HEAD_SIZE == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()
                ew = (-torch.exp(w.float())).contiguous()
                ctx.save_for_backward(r, k, v, ew, u)
                y = torch.empty((B, T, C), device=r.device, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                assert gy.dtype == torch.float32
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, ew, u = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu)

    def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
        return WKV_6.apply(B, T, C, H, r, k, v, w, u)

elif 'x052' in os.environ["RWKV_MY_TESTING"]:
    cuda_file = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv5_cuda.cu"
    cpp_file = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv5_op.cpp"
    os.makedirs(f"{os.path.dirname(__file__)}/rwkv_cuda/build", exist_ok=True)
    
    wkv5_cuda = load(name="wkv5", sources=[cuda_file, cpp_file],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"],
                    build_directory=f"{os.path.dirname(__file__)}/rwkv_cuda/build")
        
    class WKV_5(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u):
            with torch.no_grad():
                assert r.dtype == torch.float32
                assert k.dtype == torch.float32
                assert v.dtype == torch.float32
                assert w.dtype == torch.float32
                assert u.dtype == torch.float32
                assert HEAD_SIZE == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()
                ew = (-torch.exp(w.float())).contiguous()
                eew = (torch.exp(ew)).contiguous()
                ctx.save_for_backward(r, k, v, eew, ew, u)
                y = torch.empty((B, T, C), device=r.device, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                assert gy.dtype == torch.float32
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, eew, ew, u = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
                gw = torch.sum(gw, 0).view(H, C//H)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu)

    def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):
        return WKV_5.apply(B, T, C, H, r, k, v, w, u)

elif 'mamba' in os.environ["RWKV_MY_TESTING"]:
    from mamba_ssm import Mamba

########################################################################################################

class RWKV_Tmix_x052(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        assert HEAD_SIZE == self.head_size # change HEAD_SIZE to match args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = args.head_size_divisor

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        return r, k, v, g

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x / self.head_size_divisor).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g = self.jit_func(x)

        x = RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w=self.time_decay, u=self.time_faaaa)

        return self.jit_func_2(x, g)

class RWKV_Tmix_x060(MyModule):
    def __init__(self,
                 layer_id,
                 head_size_a,
                 dim_att,
                 n_layer,
                 n_embd,
                 head_size_divisor,
                 shift_pixel=1,
                 shift_c_gamma=1/4,):
        super().__init__()
        self.layer_id = layer_id

        self.head_size = head_size_a
        self.n_head = dim_att // head_size_a
        self.shift_pixel = shift_pixel
        self.shift_c_gamma = shift_c_gamma
        
        assert dim_att % self.n_head == 0, f'get `dim_att` = {dim_att} and `n_head`= {self.n_head}'

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, n_embd, 1)
            for i in range(n_embd):
                ddd[0, i, 0] = i / n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(TIME_MIX_EXTRA_DIM*5, n_embd))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(dim_att)
            for n in range(dim_att):
                decay_speed[n] = -5 + 8 * (n / (dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, dim_att, 1))

            TIME_DECAY_EXTRA_DIM = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, n_embd))
            self.time_decay_w2 = nn.Parameter(torch.zeros(dim_att, TIME_DECAY_EXTRA_DIM).uniform_(-0.01, 0.01))

            tmp = torch.zeros(dim_att)
            for n in range(dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = ShiftByConv(n_embd)  # groups_q_shift  #nn.ZeroPad2d((1, -1, 0, 0))
        self.receptance = nn.Conv1d(n_embd, dim_att, 1, groups=4)
        self.key = nn.Conv1d(n_embd, dim_att, 1, groups=4)

        self.value = nn.Conv1d(n_embd, dim_att, 1, groups=4)
        self.output = nn.Conv1d(dim_att, n_embd, 1, groups=4)
        self.gate = nn.Conv1d(n_embd, dim_att, 1, groups=4)
        self.ln_x = nn.LayerNorm(dim_att, eps=(1e-5)*(head_size_divisor**2))  # nn.GroupNorm(self.n_head, dim_att, eps=(1e-5)*(head_size_divisor**2))

    @MyFunction
    def jit_func(self, x, x_h: int, x_w: int):
        B, C, T = x.size()

        # xx = self.time_shift(x, self.shift_pixel, self.shift_c_gamma, x_h, x_w) #- x
        xx = self.time_shift(x, x_h, x_w)

        xxx = x + xx * self.time_maa_x  # [B, C, T]
        xxx = torch.tanh(self.time_maa_w1 @ xxx).view(B*T, 5, -1).transpose(0, 1)  # [5*D, C] @ [B, C, T] -> [B, 5*D, T] -> [5, B*T, D]
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1).transpose(-2, -1)  # [5, B*T, D] @ [5, D, C] -> [5, B, C, T]
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr).transpose(1, 2).contiguous()
        k = self.key(xk).transpose(1, 2).contiguous()
        v = self.value(xv).transpose(1, 2).contiguous()
        g = F.silu(self.gate(xg))
        
        # [D, C] @ [B, C, T] -> [C, D] @ [B, D, T] -> [B, C, T]
        ww = self.time_decay_w2 @ torch.tanh(self.time_decay_w1 @ xw)
        w = self.time_decay + ww  # [1, C, 1] + [B, C, T] -> [B, C, T]

        return r, k, v, g, w.transpose(1, 2).contiguous()

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x).view(B, T, C).transpose(1, 2)
        x = self.output(x * g)
        return x

    def forward(self, x, patch_resolution):
        B, C, T = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x, patch_resolution[0], patch_resolution[1])
        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.jit_func_2(x, g)

########################################################################################################

class RWKV_CMix_x052(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

class RWKV_CMix_x060(MyModule):
    def __init__(self, 
                 layer_id,
                 n_layer,
                 n_embd,
                 dim_ffn,
                 shift_pixel=1,
                 shift_c_gamma=1/4,):
        super().__init__()
        self.layer_id = layer_id
        self.shift_pixel = shift_pixel
        self.shift_c_gamma = shift_c_gamma
        self.time_shift = groups_q_shift

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, n_embd, 1)
            for i in range(n_embd):
                ddd[0, i, 0] = i / n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Conv1d(n_embd, dim_ffn, 1, groups=4)  # nn.Linear(n_embd, dim_ffn, bias=False)
        self.receptance = nn.Conv1d(n_embd, n_embd, 1, groups=4)
        self.value = nn.Conv1d(dim_ffn, n_embd, 1, groups=4)

    @MyFunction
    def jit_forward(self, x, x_h: int, x_w: int):
        xx = self.time_shift(x, self.shift_pixel, self.shift_c_gamma, x_h, x_w)
        
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return F.silu(self.receptance(xr)) * kv  #torch.sigmoid(self.receptance(xr)) * kv
    
    def forward(self, x, patch_resolution):
        
        return self.jit_forward(x, patch_resolution[0], patch_resolution[1])

########################################################################################################


class RWKV_ChannelMix(MyModule):
    def __init__(self, 
                 layer_id,
                 n_embd,
                 dim_ffn,
                 n_layer,
                 shift_pixel=1,
                 shift_c_gamma=1/4,):
        super().__init__()
        self.layer_id = layer_id
        self.shift_pixel = shift_pixel
        self.shift_c_gamma = shift_c_gamma
        self.time_shift = groups_q_shift

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        self.key = nn.Conv1d(n_embd, dim_ffn, groups=4)  # nn.Linear(n_embd, dim_ffn, bias=False)
        self.receptance = nn.Conv1d(n_embd, n_embd, groups=4)
        self.value = nn.Conv1d(dim_ffn, n_embd, groups=4)


    @MyFunction
    def jit_forward(self, x, x_h, x_w):
        xx = self.time_shift(x, self.shift_pixel, self.shift_c_gamma, x_h, x_w)
        
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv
        
    def forward(self, x, patch_resolution):
        self.jit_forward(x, patch_resolution[0], patch_resolution[1])
        

class MishGLU(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((1, -1, 0, 0))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, 
                layer_id,
                n_embd=64,
                my_pos_emb=0,
                dropout=0.,
                dim_att=64,
                dim_ffn=64,
                tiny_att_dim=0.,
                ctx_len=0,
                head_size_a=8,
                head_size_divisor=1,
                n_layer=8,
                tiny_att_layer=None,
                pre_ffn=False,
                *args,):
        super().__init__()
        # assert args is None
        
        self.pre_ffn = pre_ffn
        self.layer_id = layer_id
        self.my_pos_embd = my_pos_emb
        self.dropout = dropout
        self.tiny_att_dim = tiny_att_dim
        

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)
            if my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,my_pos_emb,n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((my_pos_emb,1,n_embd)))

        if self.layer_id == 0 and pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(0, n_embd, dim_ffn, n_layer)
        else:
            if 'x060' in os.environ["RWKV_MY_TESTING"]:
                self.att = RWKV_Tmix_x060(layer_id, head_size_a, dim_att,
                                          n_layer, n_embd, head_size_divisor)
            elif 'x052' in os.environ["RWKV_MY_TESTING"]:
                self.att = RWKV_Tmix_x052(args, layer_id)
            elif 'mamba' in os.environ["RWKV_MY_TESTING"]:
                self.att = Mamba(d_model=n_embd, d_state=16, d_conv=4, expand=2.125) # match rwkv6 #params

        if 'g' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = MishGLU(args, layer_id)
        elif 'x060' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = RWKV_CMix_x060(layer_id, n_layer, n_embd, dim_ffn)
        elif 'x052' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = RWKV_CMix_x052(args, layer_id)
        elif 'mamba' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = Mamba(d_model=n_embd, d_state=16, d_conv=4, expand=2.125) # match rwkv6 #params
        
        if tiny_att_dim > 0 and self.layer_id == tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(n_embd)
            self.tiny_q = nn.Linear(n_embd, tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(n_embd, tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(n_embd, n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(ctx_len, ctx_len)))

        if dropout > 0:
            self.drop0 = nn.Dropout(p = dropout)
            self.drop1 = nn.Dropout(p = dropout)
        
    def forward(self, x, x_emb=None):
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if self.my_pos_embd > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                x = x + pos_emb

        if self.dropout == 0:
            if self.layer_id == 0 and self.pre_ffn > 0:
                x = x + self.ffnPre(self.ln1(x))
            else:
                x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            if self.layer_id == 0 and self.pre_ffn > 0:
                x = self.drop0(x + self.ffnPre(self.ln1(x)))
            else:
                x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))

        if self.tiny_att_dim > 0 and self.layer_id == self.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (self.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
        return x
    
if __name__ == '__main__':
    from types import SimpleNamespace
    
    args = SimpleNamespace(
        n_embd=64,
        my_pos_emb=0,
        dropout=0.,
        dim_att=64,
        dim_ffn=64,
        tiny_att_dim=0.,
        ctx_len=0,
        head_size_a=8,
        head_size_divisor=1,
        n_layer=8,
    )
    
    block = Block(
        layer_id=0,
        n_embd=64,
        my_pos_emb=0,
        dropout=0.,
        dim_att=64,
        dim_ffn=64,
        tiny_att_dim=0.,
        ctx_len=0,
        head_size_a=8,
        head_size_divisor=1,
        n_layer=8
    ).cuda()
    
    x = torch.randn(1, 128*128, 64).cuda()#.float32()
    
    print(block(x).shape)
    
    
    
    