import numpy as np
from model.module.layer_norm import NAFLayerNorm
import math
from typing import Tuple, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.module.helper_func import exists, default, padding_to_multiple_of
from timm.models.layers import trunc_normal_, to_2tuple, DropPath

from model.module.swin import window_reverse, window_partition, Mlp
from model.module.resblock import DWConv
from model.module.layer_norm import normalization
from utils import exists, default


class Attention(nn.Module):
    """
    valinna attention
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, qk_scale=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        # self.scale = qk_scale if qk_scale is not None else dim_head ** -0.5
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((heads, 1, 1))), requires_grad=True
        )

        self.attend = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        logit_scale = torch.clamp(
            self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01)).to(q.device)
        ).exp()
        dots = torch.matmul(q, k.transpose(-1, -2)) * logit_scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ReducedAttention(nn.Module):
    """
    reduced k, v number tokens
    """

    def __init__(
        self,
        dim1: int,
        dim2: int,
        num_heads: int,
        bias: bool = False,
        attn_drop: float = 0.0,
        reduce_ratio: int = 4,
    ):
        super(ReducedAttention, self).__init__()
        self.match_c = nn.Conv2d(dim2, dim1, kernel_size=1, stride=1)

        self.reduce_ratio = reduce_ratio
        # self.reduce = nn.Conv2d(dim1, dim1, 5, 4, 5 // 2)
        # self.reduce = nn.MaxPool2d(kernel_size=5, stride=4, padding=5 // 2)
        self.reduce = nn.AvgPool2d(kernel_size=5, stride=reduce_ratio, padding=5 // 2)

        self.bias = bias
        self.attn_drop_prob = attn_drop

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim1, dim1, kernel_size=1, bias=bias)
        self.q_dw = nn.Conv2d(
            dim1, dim1, kernel_size=3, stride=1, padding=1, groups=dim1, bias=bias
        )
        self.kv = nn.Conv2d(dim1, dim1 * 2, kernel_size=1, bias=bias)
        self.kv_dw = nn.Conv2d(
            dim1 * 2,
            dim1 * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim1 * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(dim1, dim1, kernel_size=1, bias=bias)
        self.attn_drop = (
            nn.Dropout(self.attn_drop_prob)
            if self.attn_drop_prob != 0.0
            else nn.Identity()
        )

        # self.ghost_branch = nn.Sequential(
        #     nn.Conv2d(dim1, dim1 * 2, 1, 1),
        #     nn.Conv2d(dim1 * 2, dim1 * 2, (1, 5), stride=1, padding=(0, 2), groups=dim1 * 2),
        #     nn.BatchNorm2d(dim1 * 2),
        #     nn.GELU(),
        #     nn.Conv2d(dim1 * 2, dim1 * 2, (5, 1), stride=1, padding=(2, 0), groups=dim1 * 2),
        #     nn.BatchNorm2d(dim1 * 2),
        #     nn.GELU(),
        #     nn.Conv2d(dim1 * 2, dim1, 1, 1)
        # )

        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            # nn.init.trunc_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tgt, mem):
        b, c, h, w = tgt.shape

        interp_mem = self.match_c(F.interpolate(mem, size=tgt.shape[-2:]))
        # residual = self.reduce(tgt - interp_mem)
        # residual = self.reduce(interp_mem - tgt)
        residual = F.adaptive_avg_pool2d(
            interp_mem - tgt, output_size=tgt.shape[-1] // self.reduce_ratio
        )

        q = self.q_dw(self.q(tgt))
        kv = self.kv_dw(self.kv(residual))
        k, v = kv.chunk(2, dim=1)

        # propagate residual information to tgt
        q = rearrange(q, "b (head c) h w -> b head (h w) c", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head (h w) c", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head (h w) c", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [b, h, h1w1, h2w2]
        attn = attn.softmax(dim=-1)

        out = self.attn_drop(attn) @ v  # [b, h, c, h1w1]

        out = rearrange(
            out, "b head (h w) c -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        # @project_out aren't used in pansharpening task
        out = self.project_out(out)
        return out + tgt


class SpatialIncepAttention(nn.Module):
    """
    modified from https://github.com/shendu0321/IncepFormer/blob/master/mmseg/models/backbones/ipt.py
    """

    def __init__(
        self,
        dim1,
        dim2,
        embed_dim,
        num_heads,
        stride=5,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        # self.down_ratio = down_ratio
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.match_c = nn.Conv2d(dim2, dim1, kernel_size=1, stride=1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=(1, stride),
                stride=(1, stride),
                padding=(1, stride // 2),
                groups=embed_dim,
            ),
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=(stride, 1),
                stride=(stride, 1),
                padding=(1, stride // 2),
                groups=embed_dim,
            )
            # norm_layer(embed_dim),
            # act_layer()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=stride,
                stride=stride,
                padding=stride // 2,
                groups=embed_dim,
            ),
            # nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        )

        self.dwConv = DWConv(embed_dim, 3, 1)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, tgt, mem):
        # B C H W
        B, C, H, W = tgt.shape
        N = H * W

        mem = self.match_c(mem)
        mem = F.interpolate(mem, size=tgt.shape[-2:])
        residual = mem - tgt
        reduced_res = F.adaptive_avg_pool2d(residual, output_size=(H // 4, W // 4))

        x = tgt
        x_layer = x.reshape(B, C, -1).permute(0, 2, 1)
        q = (
            self.q(self.q_norm(x_layer))
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # [b, h, n, c]

        x = reduced_res
        x_1 = self.conv1(x).view(B, C, -1)  # 1xk kx1
        x_2 = self.conv2(x).view(B, C, -1)  # kxk
        # x_3 = F.adaptive_avg_pool2d(x, (H // 4, W // 4))
        x_3 = self.dwConv(x).view(B, C, -1)  # dwconv, shape: [b, c, n]
        x_ = torch.cat([x_1, x_2, x_3], dim=2)
        x_ = self.kv_norm(x_.permute(0, 2, 1))  # [b, n, c]
        kv = (
            self.kv(x_)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # [2, b, h, n, c]

        k, v = kv[0], kv[1]  # [b, h, n, c]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, h, n, n]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)  # [b, n, c]

        x = self.proj(x)  # [b, n, c]
        x = self.proj_drop(x)
        return x.permute(0, 2, 1).reshape(B, C, H, W) + tgt


class SGA(nn.Module):
    def __init__(self, dim, num_heads, bias=False, attn_drop=0.0):
        super(SGA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop_prob = attn_drop

        self.qv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.dw = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )
        self.temperature = 1 / math.sqrt(dim)

        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = (
            nn.Dropout(self.attn_drop_prob)
            if self.attn_drop_prob != 0.0
            else nn.Identity()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        k, q = x, x
        v = self.dw(self.qv(x))
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        with torch.no_grad():
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
        out = self.attn_drop(attn) @ v
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.proj_out(out)
        return out


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SCAM(nn.Module):
    """
    Stereo Cross Attention Module (SCAM)
    copy from https://github.com/megvii-research/NAFNet/blob/50cb1496d630dbc4165e0ff8f4b5893ca5fa00a1/basicsr/models/archs/NAFSSR_arch.py#L25

    this is much like cross-attention module, but more
    efficient
    """

    def __init__(self, tgt_c, mem_c):
        super().__init__()
        c = tgt_c
        self.scale = c**-0.5

        self.match_c = nn.Conv2d(mem_c, c, kernel_size=1, stride=1, padding=0)

        self.norm_l = NAFLayerNorm(c)
        self.norm_r = NAFLayerNorm(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.fusion_proj = nn.Conv2d(2 * c, c, 1, 1, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_l, x_r):
        # input: different scales(resolution)
        # output size is equal to x_l

        x_r = self.match_c(x_r)  # [b, tgt_c, mem_h, mem_w]
        x_r = nn.functional.interpolate(
            x_r, size=x_l.shape[-2:], mode="bilinear", align_corners=True
        )  # [b, tgt_c, tgt_h, tgt_w]

        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(
            0, 2, 1, 3
        )  # B, H, c, W (transposed)

        # NOTE: conduct attn only horizontal(vertical) direction, can we use both?
        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(
            torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l
        )  # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma

        x = torch.cat([x_l + F_r2l, x_r + F_l2r], dim=1)
        x = self.fusion_proj(x)  # [b, tgt_c, tgt_h, tgt_w]
        return x


class NAFBlock(nn.Module):
    """
    copy from https://github.com/megvii-research/NAFNet/blob/50cb1496d630dbc4165e0ff8f4b5893ca5fa00a1/basicsr/models/archs/NAFNet_arch.py#L27

    it is like in-scale(self) attention module, but
    more efficient and memory friendly, because token
    mixer is simple gate rather than softmax attention
    """

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = NAFLayerNorm(c)
        self.norm2 = NAFLayerNorm(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inp):
        x = inp

        # prenorm
        x = self.norm1(x)
        # 1x1 and dwconv
        x = self.conv1(x)
        x = self.conv2(x)
        # simple gate: x1, x2 = chunk(x); x = x1 * x2
        x = self.sg(x)
        # simple channel attn
        # x: [b, c//2, h, w] mul [b, c//2, h, w]
        x = x * self.sca(x)
        # up channel to c
        x = self.conv3(x)
        x = self.dropout1(x)
        # shortcut
        y = inp + x * self.beta

        # simple ffn
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


# Multi-DConv Head Transposed Self-Attention (MDTA)
class CAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=False, attn_drop=0.0):
        super(CAttention, self).__init__()
        print(f'CAttention: dim {dim}, num_heads {num_heads}')
        self.dim = dim
        self.bias = bias
        self.attn_drop_prob = attn_drop

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(
        #     dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=1, bias=bias
        # )  # groups can also be dim * 3
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = (
            nn.Dropout(self.attn_drop_prob)
            if self.attn_drop_prob != 0.0
            else nn.Identity()
        )
        
    def forward(self, x):
        b, c, h, w = x.shape

        # qkv = self.qkv_dwconv(self.qkv(x))
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = self.attn_drop(attn) @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out

    def flops(self, *args):

        inp = args[0]

        dim = inp.size(1)
        batch_size, _, H, W = inp.size()
        if hasattr(self, "window_size"):
            window_size = self.window_size
            N = window_size ** 2
            num_patches = H * W // N
        else:
            num_patches = 1
            window_size = 1
            N = H * W

        # window_size = module.window_size
        num_heads = self.num_heads
        # calculate flops for 1 window with token length of N
        flops = 0
        # flops += N * dim * 3 * dim
        # attn = (q @ k.transpose(-2, -1)) b head c (h w) b head (h w) c
        flops += num_heads * (dim // num_heads) * (dim // num_heads) * N
        #  x = (attn @ v)   b head c c  b head c (h w)
        flops += num_heads * N * (dim // num_heads) * (dim // num_heads)
        return num_patches * flops

# Multi-DConv Head Transposed Self-Attention (MDTA)
class CAttentionLegacy(nn.Module):
    def __init__(self, dim, num_heads, bias=False, attn_drop=0.0):
        super(CAttentionLegacy, self).__init__()
        self.dim = dim
        self.bias = bias
        self.attn_drop_prob = attn_drop

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=1, bias=bias
        )  # groups can also be dim * 3
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = (
            nn.Dropout(self.attn_drop_prob)
            if self.attn_drop_prob != 0.0
            else nn.Identity()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = self.attn_drop(attn) @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out
    
class MultiScaleWindowCrossAttention(nn.Module):
    # window_dict_train_reduce = {128: 16, 64: 8, 32: 4}
    window_dict_train_reduce =  {128: 16, 64: 8, 16: 2}
    # window_dict_test_reduce = {128: 16, 64: 8, 32: 4}
    window_dict_test_reduce = {512: 16, 256: 8, 128: 4}
    window_dict_test_full_p512 = {512: 16, 256: 8, 128: 4}
    window_dict_test_full_p256 = {256: 16, 128: 8, 64: 4}
    window_dict_test_full_p128 = {128: 16, 64: 8, 32: 4}
    window_dict_test_full_p1000 = {1000: 20, 500: 10, 250: 5}
    window_dict = window_dict_train_reduce

    def __init__(
        self,
        dim1,
        dim2,
        window_size1=None,
        window_size2=None,
        num_heads=8,
        bias=True,
        attn_drop=0.0,
        norm_type="bn",
        window_dict=None,
    ):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        print(f"mwsa {dim2}->{dim1}")

        self.window_size1 = window_size1  # Wh, Ww
        self.window_size2 = window_size2
        self.num_heads = num_heads
        self.attn_drop_prob = attn_drop
        if exists(window_dict):
            self.window_dict = window_dict

        # self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        # self.cpb_mlp = nn.Sequential(nn.Linear(2, 256, bias=True),
        #                              nn.ReLU(inplace=True),
        #                              nn.Linear(256, num_heads, bias=False))

        # proj
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.match_c = nn.Conv2d(dim2, dim1, kernel_size=1, stride=1)
        self.q = nn.Linear(dim1, dim1, bias=bias)
        self.kv = nn.Linear(dim1, dim1 * 2, bias=bias)

        self.project_out = nn.Linear(dim1, dim1)
        self.attn_drop = (
            nn.Dropout(self.attn_drop_prob)
            if self.attn_drop_prob != 0.0
            else nn.Identity()
        )

        # TODO mlp_ratio=4就不用ghost_module
        self.ghost_module = nn.Sequential(
            nn.Conv2d(
                dim1 * 2,
                dim1 * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=dim1 * 2,
                bias=False,
            ),
            SimpleGate(),
            #.BatchNorm2d(dim1),
            normalization(norm_type, dim1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                dim1,
                dim1,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=dim1,
            ),
        )

    def register_table_position_buffer(self, window_size_hw: tuple):
        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(window_size_hw[0] - 1), window_size_hw[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(window_size_hw[1] - 1), window_size_hw[1], dtype=torch.float32
        )
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2
        relative_coords_table[:, :, :, 0] /= window_size_hw[0] - 1
        relative_coords_table[:, :, :, 1] /= window_size_hw[1] - 1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / np.log2(8)
        )

        self.register_buffer(f"relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size_hw[0])
        coords_w = torch.arange(window_size_hw[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size_hw[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_hw[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size_hw[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer(f"relative_position_index", relative_position_index)

    def forward(self, tgt, mem):
        # if not exists(self.window_size1) and not exists(self.window_size2):
        #     # bug: train/test window_size1/2 is different
        #     # this 'if' will cause self.window_size1(2) not update
        #     self.window_size1 = self.window_dict[tgt.size(-1)]
        #     self.window_size2 = self.window_dict[mem.size(-1)]
        self.window_size1 = self.window_dict[int(tgt.size(-1))]
        self.window_size2 = self.window_dict[int(mem.size(-1))]

        
        b, c, h, w = tgt.shape
        mem = self.match_c(mem)
        q = window_partition(
            tgt.permute(0, 2, 3, 1), self.window_size1
        )  # [nw*b, wh1, ww1, c]
        kv = window_partition(
            mem.permute(0, 2, 3, 1), self.window_size2
        )  # [nw*b, wh2, ww2, c]

        q = self.q(q)
        kv = self.kv(kv)
        k, v = kv.chunk(2, dim=-1)

        # assert tgt.size(0) == mem.size(0)

        # q: [b*nw, nh, wh1*ww1, c]
        # k, v: [b*nw, nh, wh2*ww2, c]
        q = rearrange(q, "b h w (head c) -> b head (h w) c", head=self.num_heads)
        k = rearrange(k, "b h w (head c) -> b head (h w) c", head=self.num_heads)
        v = rearrange(v, "b h w (head c) -> b head (h w) c", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # [b*nw, nh, wh1*ww1, wh2*ww2]
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # logit_scale = torch.clamp(self.logit_scale,
        #                           max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
        # attn = attn * logit_scale

        # relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        # relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size1 * self.window_size2, self.window_size1 * self.window_size2, -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        # attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(-1)
        out = self.attn_drop(attn) @ v  # [b*nw, nh, wh1*ww1, c]

        # [b*nw, nh, nw, c]
        out = rearrange(
            out,
            "b head (h w) c -> b h w (head c)",
            head=self.num_heads,
            h=self.window_size1,
            w=self.window_size1,
        )
        out = self.project_out(out)
        out = window_reverse(out, self.window_size1, h, w).permute(0, -1, 1, 2)

        ghost_out = self.ghost_module(
            torch.cat([tgt, F.interpolate(mem, tgt.shape[-2:], mode="bilinear")], dim=1)
        )
        return out + ghost_out

    def flops(self, *args):
        # B, C, H, W
        tgt, mem = args[:-1]
        _, dim1, H1, W1 = tgt.size() # q
        _, dim2, H2, W2 = mem.size() # kv
        num_heads = self.num_heads
        L1 = self.window_size1 ** 2
        L2 = self.window_size2 ** 2
        num_patches = H1 * W1 // L1

        flops = 0
        # q: tgt 1, 128, 16, 16 -> 4*4*1, 8, 4*4, 16 (w1*w1*N, n_h, p1*p1, C1')
        # k: mem 1, 128, 64, 64 -> 4*4, 8, 16*16, 16 (w2*w2*N, n_h, p2*p2, C2')
        # 16, 8, 16, 16 x 16, 8, 16, 256 -> 16, 8, 16, 256
        # attn = (q @ k.transpose(-2, -1))
        flops += num_heads * (dim1 // num_heads) * L1 * L2
        #  x = (attn @ v) 16, 8, 16, 256 x 16, 8, 16, 256 -> 16, 8, 16, 256
        flops += num_heads * (dim2 // num_heads) * L1 * L2

        return num_patches * flops
    
"""
legacy code of spatial attention
"""
class MultiScaleWindowCrossAttentionLegacy(nn.Module):
    # window_dict_train_reduce = {128: 16, 64: 8, 16: 2}
    window_dict_train_reduce = {64: 32, 32: 16, 16: 8}
    window_dict_test_reduce = {128: 16, 64: 8, 32: 4}
    window_dict_test_full_p512 = {512: 16, 256: 8, 64: 2}
    window_dict_test_full_p256 = {256: 16, 128: 8, 64: 4}
    window_dict_test_full_p128 = {128: 16, 64: 8, 32: 4}
    window_dict_test_full_p1000 = {1000: 40, 500: 20, 125: 5}
    window_dict = window_dict_test_full_p512

    def __init__(
        self,
        dim1,
        dim2,
        window_size1=None,
        window_size2=None,
        num_heads=8,
        bias=True,
        attn_drop=0.0,
        window_dict=None,
    ):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        print(f"mwsa {dim1}->{dim2}")

        self.window_size1 = window_size1  # Wh, Ww
        self.window_size2 = window_size2
        self.num_heads = num_heads
        self.attn_drop_prob = attn_drop
        if exists(window_dict):
            self.window_dict = window_dict

        # self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        # self.cpb_mlp = nn.Sequential(nn.Linear(2, 256, bias=True),
        #                              nn.ReLU(inplace=True),
        #                              nn.Linear(256, num_heads, bias=False))

        # proj
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.match_c = nn.Conv2d(dim2, dim1, kernel_size=1, stride=1)
        self.q = nn.Linear(dim1, dim1, bias=bias)
        self.kv = nn.Linear(dim1, dim1 * 2, bias=bias)

        self.project_out = nn.Linear(dim1, dim1)
        self.attn_drop = (
            nn.Dropout(self.attn_drop_prob)
            if self.attn_drop_prob != 0.0
            else nn.Identity()
        )

        self.ghost_module = nn.Sequential(
            nn.Conv2d(
                dim1 * 2,
                dim1 * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=dim1 * 2,
                bias=False,
            ),
            SimpleGate(),
            nn.BatchNorm2d(dim1),
            nn.ReLU(),
            nn.Conv2d(
                dim1,
                dim1,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=dim1,
            ),
        )

    def register_table_position_buffer(self, window_size_hw: tuple):
        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(window_size_hw[0] - 1), window_size_hw[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(window_size_hw[1] - 1), window_size_hw[1], dtype=torch.float32
        )
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2
        relative_coords_table[:, :, :, 0] /= window_size_hw[0] - 1
        relative_coords_table[:, :, :, 1] /= window_size_hw[1] - 1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / np.log2(8)
        )

        self.register_buffer(f"relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size_hw[0])
        coords_w = torch.arange(window_size_hw[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size_hw[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_hw[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size_hw[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer(f"relative_position_index", relative_position_index)

    def forward(self, tgt, mem):
        if not exists(self.window_size1) and not exists(self.window_size2):
            self.window_size1 = self.window_dict[tgt.size(-1)]
            self.window_size2 = self.window_dict[mem.size(-1)]

        b, c, h, w = tgt.shape
        mem = self.match_c(mem)
        q = window_partition(
            tgt.permute(0, 2, 3, 1), self.window_size1
        )  # [nw*b, wh1, ww1, c]
        kv = window_partition(
            mem.permute(0, 2, 3, 1), self.window_size2
        )  # [nw*b, wh2, ww2, c]

        q = self.q(q)
        kv = self.kv(kv)
        k, v = kv.chunk(2, dim=-1)

        # assert tgt.size(0) == mem.size(0)

        # q: [b*nw, nh, wh1*ww1, c]
        # k, v: [b*nw, nh, wh2*ww2, c]
        q = rearrange(q, "b h w (head c) -> b head (h w) c", head=self.num_heads)
        k = rearrange(k, "b h w (head c) -> b head (h w) c", head=self.num_heads)
        v = rearrange(v, "b h w (head c) -> b head (h w) c", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # [b*nw, nh, wh1*ww1, wh2*ww2]
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # logit_scale = torch.clamp(self.logit_scale,
        #                           max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
        # attn = attn * logit_scale

        # relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        # relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size1 * self.window_size2, self.window_size1 * self.window_size2, -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        # attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(-1)
        out = self.attn_drop(attn) @ v  # [b*nw, nh, wh1*ww1, c]

        # [b*nw, nh, nw, c]
        out = rearrange(
            out,
            "b head (h w) c -> b h w (head c)",
            head=self.num_heads,
            h=self.window_size1,
            w=self.window_size1,
        )
        out = self.project_out(out)
        out = window_reverse(out, self.window_size1, h, w).permute(0, -1, 1, 2)

        ghost_out = self.ghost_module(
            torch.cat([tgt, F.interpolate(mem, tgt.shape[-2:], mode="bilinear")], dim=1)
        )
        return out + ghost_out


class CrossCAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads, bias=False, attn_drop=0.0):
        super(CrossCAttention, self).__init__()
        # self.dim = dim
        self.bias = bias
        self.attn_drop_prob = attn_drop

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim1, dim1, kernel_size=1, bias=bias)
        self.q_dw = nn.Conv2d(
            dim1, dim1, kernel_size=3, stride=1, padding=1, groups=dim1, bias=bias
        )
        self.kv = nn.Conv2d(dim2, dim2 * 2, kernel_size=1, bias=bias)
        self.kv_dw = nn.Conv2d(
            dim2 * 2,
            dim2 * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim2 * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(dim1, dim1, kernel_size=1, bias=bias)
        self.attn_drop = (
            nn.Dropout(self.attn_drop_prob)
            if self.attn_drop_prob != 0.0
            else nn.Identity()
        )

    def forward(self, tgt, mem):
        # tgt: [b, c1, h, w]
        # mem: [b, c2, h, w]
        # c2 = c1//(2^n) or c2 = c1*(2^n)
        b, c, h, w = tgt.shape

        q = self.q_dw(self.q(tgt))
        kv = self.kv_dw(self.kv(mem))
        k, v = kv.chunk(2, dim=1)
        # q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [b, h, c1, c2]
        attn = attn.softmax(dim=-1)

        out = self.attn_drop(attn) @ v  # [b, h, c1, hw]

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


# copy from https://github.com/megvii-research/TLC/blob/main/basicsr/models/archs/restormer_arch.py
# convert CAttention in Restormer to Local Attention
# diminish train-inference gap
class LocalAttention(CAttention):
    def __init__(
        self,
        dim,
        num_heads,
        bias,
        attn_drop,
        base_size=None,
        kernel_size=None,
        fast_imp=False,
        train_size=None,
    ):
        super().__init__(dim, num_heads, bias, attn_drop=attn_drop)
        self.base_size = base_size
        self.kernel_size = kernel_size
        self.fast_imp = fast_imp
        self.train_size = train_size

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 3, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math

        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i : i + k1, j : j + k2])
                idxes.append({"i": i, "j": j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx["i"]
            j = each_idx["j"]
            preds[0, :, i : i + k1, j : j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i : i + k1, j : j + k2] += 1.0

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def _forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out

    def _pad(self, x):
        b, c, h, w = x.shape
        k1, k2 = self.kernel_size
        mod_pad_h = (k1 - h % k1) % k1
        mod_pad_w = (k2 - w % k2) % k2
        pad = (
            mod_pad_w // 2,
            mod_pad_w - mod_pad_w // 2,
            mod_pad_h // 2,
            mod_pad_h - mod_pad_h // 2,
        )
        x = F.pad(x, pad, "reflect")
        return x, pad

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        if self.fast_imp:
            raise NotImplementedError
            # qkv, pad = self._pad(qkv)
            # b,C,H,W = qkv.shape
            # k1, k2 = self.kernel_size
            # qkv = qkv.reshape(b,C,H//k1, k1, W//k2, k2).permute(0,2,4,1,3,5).reshape(-1,C,k1,k2)
            # out = self._forward(qkv)
            # out = out.reshape(b,H//k1,W//k2,c,k1,k2).permute(0,3,1,4,2,5).reshape(b,c,H,W)
            # out = out[:,:,pad[-2]:pad[-2]+h, pad[0]:pad[0]+w]
        else:
            qkv = self.grids(qkv)  # convert to local windows
            out = self._forward(qkv)
            out = rearrange(
                out,
                "b head c (h w) -> b (head c) h w",
                head=self.num_heads,
                h=qkv.shape[-2],
                w=qkv.shape[-1],
            )
            out = self.grids_inverse(out)  # reverse

        out = self.project_out(out)
        return out


# class CrossAttention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0., qk_scale=None):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)
#
#         self.heads = heads
#         self.scale = qk_scale if qk_scale is not None else dim_head ** -0.5
#
#         self.attend = nn.Softmax(dim=-1)
#
#         self.dropout = nn.Dropout(dropout)
#         self.q = nn.Linear(dim, inner_dim, bias=False)
#         self.k = nn.Linear(dim, inner_dim, bias=False)
#         self.v = nn.Linear(dim, inner_dim, bias=False)
#
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#
#     def forward(self, tgt, mem):
#         # for the first layer
#         # tgt is pan, mem is ms
#         q, k, v = self.q(tgt), self.k(mem), self.v(mem)
#
#         # qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])
#
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#
#         attn = self.attend(dots)
#         attn = self.dropout(attn)
#
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, dim2, num_heads, bias=False, attn_drop=0.0):
        super(CrossAttention, self).__init__()
        # self.dim = dim
        self.bias = bias
        self.attn_drop_prob = attn_drop

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dw = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )
        self.kv = nn.Conv2d(dim2, dim * 2, kernel_size=1, bias=bias)
        self.kv_dw = nn.Conv2d(
            dim * 2,
            dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = (
            nn.Dropout(self.attn_drop_prob)
            if self.attn_drop_prob != 0.0
            else nn.Identity()
        )

    def forward(self, tgt, mem):
        # tgt: [b, c, h1, w1]
        # mem: [b, c2, h2, w2]
        # attn: [b, h1w1, h2w2] regardless of head
        b, c, h, w = tgt.shape

        q = self.q_dw(self.q(tgt))
        kv = self.kv_dw(self.kv(mem))
        k, v = kv.chunk(2, dim=1)
        # q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head (h w) c", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head (h w) c", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head (h w) c", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [b, h, h1w1, h2w2]
        attn = attn.softmax(dim=-1)

        out = self.attn_drop(attn) @ v  # [b, h, c, h1w1]

        out = rearrange(
            out, "b head (h w) c -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = torch.einsum("... d, h d -> ... h d", x, self.gamma) + self.beta
        return out.unbind(dim=-2)


class GAU(nn.Module):
    def __init__(
        self,
        *,
        dim,
        query_key_dim=128,
        expansion_factor=2.0,
        add_residual=True,
        causal=False,
        dropout=0.0,
        norm_klass=nn.LayerNorm,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)

        self.norm = norm_klass(dim)
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = nn.Sequential(nn.Linear(dim, hidden_dim * 2), nn.SiLU())

        self.to_qk = nn.Sequential(nn.Linear(dim, query_key_dim), nn.SiLU())

        self.offsetscale = OffsetScale(query_key_dim, heads=2)

        self.to_out = nn.Sequential(nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

        self.add_residual = add_residual

    def forward(self, x, rel_pos_bias=None):
        """
        gate: $\mathbf U=\Phi_u(\mathbf X \mathbf W_u)$
        v: $\mathbf V=\Phi_v(\mathbf X \mathbf W_v)$
        offsetscale output: $\mathbf Z=\Phi_z(\mathbf X \mathbf W_z)$
        q: $\mathcal Q(\mathbf Z)$
        k: $\mathcal K(\mathbf Z)$
        attn: $\mathbf A=\cfrac 1n \text{ReLU}^2\left(\cfrac{\mathcal Q(\mathbf Z) \mathcal K(\mathbf Z)^T}{\sqrt s}\right)$
        out: $\mathbf O=(\mathbf U \odot \mathbf{AV})\mathbf W_o$

        Note that for token expansion, we change the $relu^2$ to Softmax, and in paper, only one head is enough.
        :param x:
        :param rel_pos_bias:
        :param mask:
        :return:
        """
        seq_len, device = x.shape[-2], x.device

        normed_x = self.norm(x)
        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)

        qk = self.to_qk(normed_x)
        q, k = self.offsetscale(qk)

        sim = torch.einsum("... i d, ... i D -> ... d D", q, k) / seq_len

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        # attn = F.relu(sim) ** 2
        attn = sim.softmax(-1)
        attn = self.dropout(attn)

        # if exists(mask):
        #     mask = rearrange(mask, 'b j -> b 1 j')
        #     attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones(
                (seq_len, seq_len), dtype=torch.bool, device=device
            ).triu(1)
            attn = attn.masked_fill(causal_mask, 0.0)

        out = torch.einsum("... d D, ... j d -> ... j d", attn, v)
        out = out * gate

        out = self.to_out(out)

        if self.add_residual:
            out = out + x

        return out


class FLASH(nn.Module):
    def __init__(
        self,
        *,
        dim,
        group_size=256,
        query_key_dim=128,
        expansion_factor=2.0,
        causal=False,
        dropout=0.0,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens=False,
        reduce_group_non_causal_attn=True,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # positional embeddings

        self.rotary_pos_emb = rotary_pos_emb
        # self.rel_pos_bias = T5RelativePositionBias(query_key_dim ** 0.5, causal=causal)

        # norm

        self.norm = norm_klass(dim)
        self.dropout = nn.Dropout(dropout)

        # whether to reduce groups in non causal linear attention

        self.reduce_group_non_causal_attn = reduce_group_non_causal_attn

        # projections

        self.to_hidden = nn.Sequential(nn.Linear(dim, hidden_dim * 2), nn.SiLU())

        self.to_qk = nn.Sequential(nn.Linear(dim, query_key_dim), nn.SiLU())

        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, *, mask=None):
        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        # prenorm

        normed_x = self.norm(x)

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.0)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        # initial projections

        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)

        # offset and scale

        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)

        # mask out linear attention keys

        if exists(mask):
            lin_mask = rearrange(mask, "... -> ... 1")
            lin_k = lin_k.masked_fill(~lin_mask, 0.0)

        # rotate queries and keys

        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(
                self.rotary_pos_emb.rotate_queries_or_keys,
                (quad_q, lin_q, quad_k, lin_k),
            )

        # padding for groups

        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v = map(
                lambda t: F.pad(t, (0, 0, 0, padding), value=0.0),
                (quad_q, quad_k, lin_q, lin_k, v),
            )

            mask = default(mask, torch.ones((b, n), device=device, dtype=torch.bool))
            mask = F.pad(mask, (0, padding), value=False)

        # group along sequence

        quad_q, quad_k, lin_q, lin_k, v = map(
            lambda t: rearrange(t, "b (g n) d -> b g n d", n=self.group_size),
            (quad_q, quad_k, lin_q, lin_k, v),
        )

        if exists(mask):
            mask = rearrange(mask, "b (g j) -> b g 1 j", j=g)

        # calculate quadratic attention output

        sim = torch.einsum("... i d, ... j d -> ... i j", quad_q, quad_k) / g

        # sim = sim + self.rel_pos_bias(sim)

        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.0)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.0)

        quad_out = torch.einsum("... i j, ... j d -> ... i d", attn, v)

        # calculate linear attention output

        if self.causal:
            lin_kv = torch.einsum("b g n d, b g n e -> b g d e", lin_k, v) / g

            # exclusive cumulative sum along group dimension

            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.0)

            lin_out = torch.einsum("b g d e, b g n d -> b g n e", lin_kv, lin_q)
        else:
            context_einsum_eq = (
                "b d e" if self.reduce_group_non_causal_attn else "b g d e"
            )
            lin_kv = (
                torch.einsum(f"b g n d, b g n e -> {context_einsum_eq}", lin_k, v) / n
            )
            lin_out = torch.einsum(
                f"b g n d, {context_einsum_eq} -> b g n e", lin_q, lin_kv
            )

        # fold back groups into full sequence, and excise out padding

        quad_attn_out, lin_attn_out = map(
            lambda t: rearrange(t, "b g n d -> b (g n) d")[:, :n], (quad_out, lin_out)
        )

        # gate

        out = gate * (quad_attn_out + lin_attn_out)

        # projection out and residual

        return self.to_out(out) + x


class DynamicAttentionConv(nn.Module):
    def __init__(
        self, dim, num_heads, conv_ksize=3, stride=1, bias=False, attn_drop=0.0
    ):
        super(DynamicAttentionConv, self).__init__()
        self.dim = dim
        self.conv_ksize = conv_ksize
        self.bias = bias
        self.attn_drop_prob = attn_drop

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.proj_kernel = nn.Conv2d(dim, dim * conv_ksize**2, 1, 1)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = (
            nn.Dropout(self.attn_drop_prob)
            if self.attn_drop_prob != 0.0
            else nn.Identity()
        )

        self.unfold = nn.Unfold(conv_ksize, 1, (conv_ksize - 1) // 2, stride)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        kernel = self.attn_drop(attn) @ v

        kernel = rearrange(
            kernel, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        kernel = self.proj_kernel(kernel).view(b, c, self.conv_ksize**2, h, w)

        unfold_x = self.unfold(x).view(b, self.dim, self.conv_ksize**2, h, w)
        out = (kernel * unfold_x).sum(2)

        out = self.project_out(out)
        return out


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
        )
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / np.log2(8)
        )

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q shape: [b*nw, nh, N, c]
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01)).to(attn.device)
        ).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(
            -1, self.num_heads
        )
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, window_size={self.window_size}, "
            f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
        )

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, C, H1, W1 = x.shape
        x = x.permute(0, 2, 3, 1)

        H, W = self.input_resolution
        assert H1 == H and W1 == W, "input feature has wrong size"
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        # x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        # x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x.permute(0, 3, 1, 2)


if __name__ == "__main__":
    from fvcore.nn import flop_count_table, FlopCountAnalysis
    from thop import profile
    import thop

    # ms_token = torch.randn(1, 64 * 64, 64)
    # pan_token = torch.randn(1, 64 * 64 // 4, 64)
    # gau = GAU(dim=64)
    # flash = FLASH(dim=64, group_size=8)
    # print(gau(ms_token).shape)
    x = torch.randn(1, 3, 128, 128)
    y = torch.randn(1, 128, 16, 16)
    # ----------------FLOPs and params-------------
    # all channel is set to 64 and spatial size is (64, 64)
    # | module            | #parameters or shape   | #flops   |
    # |:------------------|:-----------------------|:---------|
    # | NAFBlock          | 30.784K                | 0.106G   |
    # | SpatialIncepBlock | 23.744K                | 0.213G   |
    # | SCAM(only 64)     | 17.024K                | 0.117G   |
    # | SCAM(128 -> 64)   | 33.534K                | 0.154G   |
    # | CAttention        | 0.348M                 | 1.43G    |

    # ---------------------------------------------

    # sga = SGA(64, 8, False, 0.1)
    # print(sga(x).shape)

    # dac = DynamicAttentionConv(64, 4)
    # print(dac(x).shape)

    # ra = ReducedAttention(64, 128, 8)
    # mwsa = MultiScaleWindowCrossAttention(64, 128)
    wsa = SwinTransformerBlock(
        3,
        window_size=8,
        shift_size=2,
        mlp_ratio=4,
        input_resolution=(128, 128),
        num_heads=1,
    )
    # nafblock = NAFBlock(64)
    # scam = SCAM(64, 128)
    # print(scam(x, y).shape)
    # icpattn = SpatialIncepAttention(64, 128, 64, 8)
    # cattn = CAttention(64, 8)
    # print(ra(x, y).shape)
    # print(cattn(x).shape)
    print(wsa(x).shape)

    # ca = CrossAttention(64, 8)
    # print(icpattn(x, y).shape)

    # print(flop_count_table(FlopCountAnalysis(gau, ms_token)))
    # print(flop_count_table(FlopCountAnalysis(flash, ms_token)))
    # print(flop_count_table(FlopCountAnalysis(icpattn, (x, y))))
    # macs, params = profile(
    #     icpattn, (x, y)
    #     )
    # macs = thop.clever_format(macs)
    # params = thop.clever_format(params)
    # print(macs, params)  # 27.84K, 0.038G
