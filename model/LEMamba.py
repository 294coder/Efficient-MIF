import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
import math
from torch import einsum
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

import sys

sys.path.append("./")
sys.path.append("../")

from model.module.vmamba_module_v3 import VSSBlock
from model.module.layer_norm import LayerNorm as LayerNorm2d
from model.base_model import BaseModel, register_model, PatchMergeModule

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum("..., f -> ... f", t, freqs)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)

        freqs_w = torch.einsum("..., f -> ... f", t, freqs)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

        print("======== shape of rope freq", self.freqs_cos.shape, "========")

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert (
            rot_dim <= t.shape[-1]
        ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        t_left, t, t_right = (
            t[..., :start_index],
            t[..., start_index:end_index],
            t[..., end_index:],
        )
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        self.freqs = freqs

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        self.seq_len = ft_seq_len

        self.alter_seq_len(ft_seq_len, pt_seq_len)

    def alter_seq_len(self, ft_seq_len, pt_seq_len):
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, self.freqs)
        # freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).cuda()
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).cuda()

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        print("======== shape of rope freq", self.freqs_cos.shape, "========")

    def forward(self, t):
        if t.shape[1] % 2 != 0:
            t_spatial = t[:, 1:, :]
            t_spatial = (
                t_spatial * self.freqs_cos + rotate_half(t_spatial) * self.freqs_sin
            )
            return torch.cat((t[:, :1, :], t_spatial), dim=1)
        else:
            return t * self.freqs_cos + rotate_half(t) * self.freqs_sin

    def __repr__(self):
        return (
            f"VisionRotaryEmbeddingFast(seq_len={self.seq_len}, freqs_cos={tuple(self.freqs_cos.shape)}, "
            + f"freqs_sin={tuple(self.freqs_sin.shape)})"
        )


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def NonLinearity(inplace=False):
    return nn.SiLU(inplace)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm2d(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, 1, 1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


def default_conv(dim_in, dim_out, kernel_size=3, bias=False):
    return nn.Conv2d(
        dim_in, dim_out, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class Block(nn.Module):
    def __init__(self, conv, dim_in, dim_out, act=NonLinearity()):
        super().__init__()
        self.proj = conv(dim_in, dim_out)
        self.act = act

    def forward(self, x, scale_shift=None):
        x = self.proj(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, conv, dim_in, dim_out, time_emb_dim=None, act=NonLinearity()):
        super(ResBlock, self).__init__()
        self.mlp = (
            nn.Sequential(act, nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim
            else None
        )

        self.block1 = Block(conv, dim_in, dim_out, act)
        self.block2 = Block(conv, dim_out, dim_out, act)
        self.res_conv = conv(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


# channel attention
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm2d(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


# self attention on each channel
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Linear(
            dim, hidden_dim * 3, bias=False
        )  # nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # b, c, h, w = x.shape
        # b, c, n = x.shape
        x = rearrange(x, "b d n -> b n d")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v]
        )

        # For 2D image
        # q, k, v = map(
        #     lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        # )

        # q = q * self.scale

        # sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        # attn = sim.softmax(dim=-1)
        # out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)

        # out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        # For 1D sequence
        q = q * self.scale

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        # v = v / n

        context = torch.einsum("b h n d, b h n e -> b h d e", k, v)
        out = torch.einsum("b h d e, b h n d -> b h e n", context, q)
        out = rearrange(out, "b h e n -> b n (h e)")
        out = self.to_out(out)

        return out.transpose(1, 2)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def initialize_weights(net_l, scale=1.0):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity='relu')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def convs(in_chan, out_chan=None, conv_type='conv3'):
    if conv_type == 'conv3':
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1),
            nn.Conv2d(out_chan, out_chan, 3, 1, 1, groups=out_chan),
        )
    elif conv_type == 'dwconv3':
        return nn.Conv2d(in_chan, in_chan, 3, 1, 1, groups=in_chan)
    elif conv_type == 'conv1':
        return nn.Conv2d(in_chan, out_chan, 1)
    else:
        raise ValueError(f'Unknown conv type {conv_type}')
    
       
class MambaInjectionBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        inner_chan,
        drop_path=0.0,
        mlp_drop=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_size=8,
        d_states=[16, 32],
        dt_rank="auto",
        ssm_conv=[3, 11],
        ssm_ratio=2,
        mlp_ratio=4,
        forward_type="v4",
        use_ckpt=False,
        local_shift_size=0,
        prev_state_chan=None,
        skip_state_chan=None,
        local_state_to_global=False,
        post_norm=False,
        **mamba_kwargs,
    ):
        super().__init__()
        self.inner_chan = inner_chan
        self.window_size = window_size
        self.local_shift_size = local_shift_size
        self.local_state_to_global = local_state_to_global
        
        # v1: intro_conv
        # self.intro_conv = convs(in_chan, inner_chan, 'conv1')
        # self.intro_to_latent = nn.Linear(inner_chan*2, inner_chan, bias=False)
        
        # v2: intro_conv
        self.intro_conv = convs(in_chan, inner_chan, 'conv3')
        
        ssm_local_conv, ssm_global_conv = ssm_conv[0], ssm_conv[1]
        local_d_state, global_d_state = d_states[0], d_states[1]
        
        # v3: mamba in mamba
        if local_d_state is not None and ssm_local_conv is not None:
            self.local_state_to_global = False
            
            self.mamba_inner_shared = VSSBlock(
                inner_chan,
                drop_path=drop_path,
                mlp_ratio=1,
                norm_layer=norm_layer,
                mlp_drop_rate=0.0,
                ssm_d_state=local_d_state,
                ssm_conv=ssm_local_conv,
                ssm_dt_rank=dt_rank,
                ssm_ratio=2,
                ssm_init='v0',
                forward_type=forward_type,
                use_checkpoint=use_ckpt,
                prev_state_chan=None,
                skip_state_chan=None,
                post_norm=post_norm,
                mlp_type='gmlp',
                # prev_state_gate=prev_state_gate,
                **mamba_kwargs,
            )
            # self.norm = norm_layer(inner_chan)
            self.enhanced_factor = nn.Parameter(torch.zeros(1, 1, 1, inner_chan), requires_grad=True)
            nn.init.trunc_normal_(self.enhanced_factor, std=0.02)
        
        self.mamba = VSSBlock(
            hidden_dim=inner_chan,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            mlp_drop_rate=mlp_drop,
            ssm_d_state=global_d_state,
            ssm_conv=ssm_global_conv,
            ssm_dt_rank=dt_rank,
            ssm_ratio=ssm_ratio,
            ssm_init='v0',
            forward_type=forward_type,
            use_checkpoint=use_ckpt,
            # prev_state_gate=False,  # may use much gpu mem
            prev_state_chan=prev_state_chan,
            skip_state_chan=skip_state_chan,
            post_norm=post_norm,
            mlp_type='gmlp',
            **mamba_kwargs,
        )
        
    def forward(self, 
                feat: torch.Tensor, 
                cond: torch.Tensor, 
                c_shuffle=True,  # decrepted
                prev_local_state: torch.Tensor=None,
                prev_global_state: torch.Tensor=None,
                skip_local_state: torch.Tensor=None,
                skip_global_state: torch.Tensor=None,
                ):
        b, h, w, c = feat.shape
        cond = F.interpolate(cond, size=(h, w), mode="bilinear", align_corners=True)

        # intro_conv v1: on cond
        cond = self.intro_conv(cond)
        cond = cond.permute(0, 2, 3, 1)
        x = feat + cond
            
        # check cache for mamba blocks
        if not self.mamba.prev_state_gate:
            prev_global_state = None
        
        if hasattr(self, 'mamba_inner_shared'):
            if not self.mamba_inner_shared.prev_state_gate: 
                prev_local_state = None
            
            # v3: mamba in mamba
            if self.local_shift_size > 0:
                x = torch.roll(x, shifts=(-self.local_shift_size, -self.local_shift_size), dims=(1, 2))            
            
            xs_local = window_partition(x, self.window_size)
            xs_local, local_ssm_state = self.mamba_inner_shared(xs_local, prev_local_state, skip_local_state)
            x_local = window_reverse(xs_local, self.window_size, h, w)
        
            if self.local_shift_size > 0:
                x_local = torch.roll(x_local, shifts=(self.local_shift_size, self.local_shift_size), dims=(1, 2))
                
            x = x_local * self.enhanced_factor + x
        else: local_ssm_state = None
  
        # local ssm state to global ssm state
        if self.local_state_to_global and self.mamba.prev_state_gate and local_ssm_state is not None:
            local_to_global_sta = reduce(local_ssm_state, '(b w) d n -> b d n', 'mean', b=b)
            if prev_global_state is not None:
                prev_global_state = local_to_global_sta + prev_global_state
            else:
                prev_global_state = local_to_global_sta
        
        # ensure to call
        x, global_ssm_state = self.mamba(x, prev_global_state, skip_global_state)
        
        return x, local_ssm_state, global_ssm_state


class UniSequential(nn.Module):
    def __init__(self, *args: tuple[nn.Module]):
        super().__init__()
        self.mods = nn.ModuleList(args)

    def __getitem__(self, idx):
        return self.mods[idx]
    
    def NAF_enc_forward(self, feat, cond):
        outp = feat
        for mod in self.mods:
            outp = mod(outp, cond)
        return outp
        
    def NAF_dec_forward(self, feat, cond):
        outp = feat
        outp = self.mods[0](outp)
        for mod in self.mods[1:]:
            outp = mod(outp, cond)
        return outp
    
    # @get_local('outp', 'global_state')
    def LEMM_enc_forward(self, 
                         feat, 
                         cond,
                         c_shuffle=False,
                         states=[None, None]):
        outp = feat
        local_state, global_state = states[0], states[1]
        for i, mod in enumerate(self.mods):
            # print(f'==encoder mods {i}')
            outp, local_state, global_state = mod(outp, cond, c_shuffle, local_state, global_state) # in_block states share
            # print(f'local_state: {local_state.shape}, global_state: {global_state.shape}')
            
        return outp, (local_state, global_state)

    def LEMM_dec_forward(self, 
                         feat, 
                         cond, 
                         c_shuffle=True, 
                         prev_states=[None, None],
                         skip_states=[None, None]):
        # skip_local_state, skip_global_state = skip_states[0], skip_states[1]
        # prev_local_state, prev_global_state = prev_states[0], prev_states[1]
        
        feat = self.mods[0](feat)
        outp = feat
        for i, mod in enumerate(self.mods[1:]):
            # print(f'==decoder mods {i}')
            outp, local_state, global_state = mod(outp, cond, c_shuffle, *(prev_states + skip_states))  # in_block states share
            # print(f'local_state: {local_state.shape}, global_state: {global_state.shape}')
            
        return outp, (local_state, global_state)
    

class SquareReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x) ** 2


############# LE-Mamba Model ################


class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2


class LEVMBlock(nn.Module):
    def __init__(
        self, c, cond_c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0
    ):
        super().__init__()

        dw_channel = c * DW_Expand
        
        self.cond_intro_conv = nn.Conv2d(cond_c, c, 1)
        
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=False,
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
            bias=False,
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
                bias=False,
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
            bias=False,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=False,
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


    def forward(self, x, cond):
        cond = F.interpolate(cond, size=x.shape[2:], mode="bilinear", align_corners=True)
        cond = self.cond_intro_conv(cond)
        x = x + cond
        
        inp = x

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x


class Permute(nn.Module):
    def __init__(self, mode="c_first"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == "c_first":
            # b h w c -> b c h w
            return x.permute(0, 3, 1, 2)
        elif self.mode == "c_last":
            # b c h w -> b h w c
            return x.permute(0, 2, 3, 1)
        else:
            raise NotImplementedError
        
    def __repr__(self):
        return f"Permute(mode={self.mode})"


def down(chan, down_type='patch_merge', permute=False, r=2, chan_r=2):
    if down_type == 'conv':
        return nn.Sequential(
            # Rearrange('b h w c -> b c h w', h=h, w=w),
            Permute("c_first") if permute else nn.Identity(),
            nn.Conv2d(chan, chan * chan_r, r, r),
            # Rearrange('b c h w -> b h w c'),
            Permute("c_last") if permute else nn.Identity(),
        )
    elif down_type == 'patch_merge':
        return PatchMerging2D(chan, chan*2)
    else:
        raise NotImplementedError(f'down type {down_type} not implemented')


def up(chan, permute=False, r=2, chan_r=2,):
    return nn.Sequential(
        # Rearrange('b h w c -> b c h w', h=h, w=w),
        Permute("c_first") if permute else nn.Identity(),
        # ver 1: use pixelshuffle
        # ver 2: directly upsample and half the channels
        nn.Conv2d(chan, chan // chan_r, 1, bias=False),
        # nn.PixelShuffle(2),
        nn.Upsample(scale_factor=r, mode="bilinear"),
        # Rearrange('b c h w -> b h w c'),
        Permute("c_last") if permute else nn.Identity(),
    )


@register_model("LEMamba")
class LEMambaNet(BaseModel):
    def __init__(
        self,
        img_channel=3,
        condition_channel=3,
        out_channel=3,
        width=16,
        # NAFBlock settings
        naf_enc_blk_nums=[],
        naf_dec_blk_nums=[],
        naf_chan_upscale=[],
        # LEMMBlock settings
        ssm_enc_blk_nums=[],
        middle_blk_nums=[],
        ssm_dec_blk_nums=[],
        ssm_enc_convs=[],
        ssm_dec_convs=[],
        ssm_ratios=[],
        ssm_chan_upscale=[],
        ssm_enc_d_states=[],
        ssm_dec_d_states=[],
        ssm_local_to_global=[],
        window_sizes=[],
        # model settings
        upscale=1,
        if_abs_pos=True,
        if_rope=False,
        pt_img_size=64,
        drop_path_rate=0.1,
        patch_merge=True,
    ):
        super().__init__()
        self.upscale = upscale
        self.if_abs_pos = if_abs_pos
        self.rope = if_rope
        self.pt_img_size = pt_img_size
        
        if len(ssm_local_to_global) == 0:
            ssm_local_to_global = [False] * len(ssm_enc_blk_nums)

        if if_abs_pos:
            self.abs_pos = nn.Parameter(torch.randn(1, pt_img_size, pt_img_size, width), requires_grad=True)

        if if_rope:
            self.rope = VisionRotaryEmbeddingFast(chan, pt_seq_len=pt_img_size, ft_seq_len=None)

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=out_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        ## main body
        if len(naf_enc_blk_nums) != 0 or len(naf_dec_blk_nums) != 0:
            assert len(naf_enc_blk_nums) == len(naf_dec_blk_nums), 'NAF encoder and decoder should have the same number of blocks'
            self.naf_encoders = nn.ModuleList()
            self.naf_decoders = nn.ModuleList()
            self.naf_ups = nn.ModuleList()
            self.naf_downs = nn.ModuleList()
            
        self.lemm_encoders = nn.ModuleList()
        self.lemm_decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.lemm_ups = nn.ModuleList()
        self.lemm_downs = nn.ModuleList()
        

        depth = sum(naf_enc_blk_nums) + sum(naf_dec_blk_nums) + middle_blk_nums + sum(ssm_enc_blk_nums) + sum(ssm_dec_blk_nums)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = dpr

        chan = width
        n_prev_blks = 0
        
        ## encoder
        # NAF layer
        for enc_i, num in enumerate(naf_enc_blk_nums):
            self.naf_encoders.append(
                UniSequential(
                    *[LEVMBlock(chan, condition_channel, drop_out_rate=inter_dpr[n_prev_blks + i]) 
                      for i in range(num)]
                )
            )
            self.naf_downs.append(down(chan, down_type='conv'))
            chan = chan * naf_chan_upscale[enc_i]
            n_prev_blks += num
            pt_img_size //= 2
            
        # LEMM layer
        print('=== init SSM encoder ===')
        for enc_i, num in enumerate(ssm_enc_blk_nums):
            
            def prev_state_chan_fn(i, only_share_in_blk=True, mod='enc'):
                if mod == 'enc':
                    if enc_i == 0:
                        if i == 0: prev_state_chan = None
                        else:
                            prev_state_chan = chan * ssm_ratios[enc_i]
                    else:
                        if i == 0:
                            if not only_share_in_blk:
                                prev_state_chan = prev_ssm_chan 
                            else: prev_state_chan = None
                        else: 
                            prev_state_chan = chan * ssm_ratios[enc_i]
                        
                elif mod == 'mid':
                    if i == 0:
                        if not only_share_in_blk:
                            prev_state_chan = prev_ssm_chan 
                        else: prev_state_chan = None
                    else:
                        prev_state_chan = chan * ssm_ratios[-1]
                
                elif mod == 'dec':
                    if (i == 0) and (not only_share_in_blk):
                        prev_state_chan = None
                    else:
                        prev_state_chan = prev_ssm_chan
                        
                # print(f'prev_state_chan={prev_state_chan * 4 if prev_state_chan is not None else prev_state_chan}')  # K=4
                return prev_state_chan
            
            
            # prev_state_chan_fn = (lambda i: None if i == 0 else chan * ssm_ratios[enc_i]) if enc_i == 0 else \
            #                      (lambda i: prev_ssm_chan if i == 0 else chan * ssm_ratios[enc_i])
            self.lemm_encoders.append(
                UniSequential(
                    *[
                        MambaInjectionBlock(
                            condition_channel,
                            chan,
                            ssm_conv=ssm_enc_convs[enc_i],
                            window_size=window_sizes[enc_i],
                            d_states=ssm_enc_d_states[enc_i],
                            ssm_ratio=ssm_ratios[enc_i],
                            drop_path=inter_dpr[n_prev_blks + i],
                            prev_state_chan=prev_state_chan_fn(i, mod='enc'),
                            local_shift_size=0,
                            local_state_to_global=ssm_local_to_global[enc_i],
                        )
                        for i in range(num)
                    ]
                )
            )
            self.lemm_downs.append(down(chan, down_type='conv', permute=True, r=2, chan_r=ssm_chan_upscale[enc_i]))
            prev_ssm_chan = chan * ssm_ratios[enc_i]
            chan = chan * ssm_chan_upscale[enc_i]
            n_prev_blks += num
            pt_img_size //= 2
            # print('==='*10)

        ## middel layer
        print('=== init SSM middle blks ===')
        # prev_state_chan_fn = lambda i: prev_ssm_chan if i == 0 else chan * ssm_ratios[-1]
        self.middle_blks = UniSequential(
            *[
                MambaInjectionBlock(
                    condition_channel,
                    chan,
                    ssm_conv=ssm_enc_convs[-1],
                    window_size=window_sizes[-1],
                    d_states=ssm_enc_d_states[-1],
                    ssm_ratio=ssm_ratios[-1],
                    drop_path=inter_dpr[n_prev_blks + i],
                    prev_state_chan=prev_state_chan_fn(i, mod='mid'),
                    local_state_to_global=ssm_local_to_global[-1],
                    # prev_state_gate=use_prev_ssm_state if i != 0 else False
                )
                for i in range(middle_blk_nums)
            ]
        )
        n_prev_blks += middle_blk_nums
        prev_ssm_chan = chan * ssm_ratios[-1]
        # print('==='*10)
        

        ## decoder
        # skip scales
        self.skip_scales = nn.ParameterList([])
        
        # LEMM layer
        print('=== init SSM decoder ===')
        for dec_i, num in enumerate(reversed(ssm_dec_blk_nums)):
            self.lemm_ups.append(up(chan, permute=True, chan_r=ssm_chan_upscale[::-1][dec_i]))
            # prev_chan_fn = lambda i: prev_ssm_chan #if i == 0 else chan * ssm_ratios[::-1][dec_i]
            chan = chan // ssm_chan_upscale[::-1][dec_i]
            self.skip_scales.append(
                nn.Parameter(torch.zeros(1, 1, 1, chan), requires_grad=True)
            )
            pt_img_size *= 2

            self.lemm_decoders.append(
                UniSequential(
                    nn.Linear(chan * 2, chan),
                    *[
                        MambaInjectionBlock(
                            condition_channel,
                            chan,
                            ssm_conv=ssm_dec_convs[dec_i],
                            window_size=window_sizes[::-1][dec_i],
                            d_states=ssm_dec_d_states[dec_i],
                            ssm_ratio=ssm_ratios[::-1][dec_i],
                            drop_path=inter_dpr[n_prev_blks + i],
                            prev_state_chan=prev_state_chan_fn(i, mod='dec'),
                            skip_state_chan=chan if i == 0 else None,  # assert skip_state_chan == chan
                            local_shift_size=0,
                            local_state_to_global=ssm_local_to_global[::-1][dec_i],
                        )
                        for i in range(num)
                    ],
                )
            )
            n_prev_blks += num
            prev_ssm_chan = chan * ssm_ratios[::-1][dec_i]
            # print('==='*10)
            
        # NAF layer
        for dec_i, num in enumerate(naf_dec_blk_nums):
            self.naf_ups.append(up(chan))
            chan = chan // naf_chan_upscale[::-1][dec_i]
            self.skip_scales.append(
                nn.Parameter(torch.zeros(1, 1, 1, chan), requires_grad=True)
            )
            pt_img_size *= 2
            self.naf_decoders.append(
                UniSequential(
                    nn.Conv2d(chan*2, chan, 1),
                    *[LEVMBlock(chan, condition_channel, drop_out_rate=inter_dpr[n_prev_blks + i]) 
                      for i in range(num)]
                )
            )
            n_prev_blks += num

        # for patch merging if the image is too large
        self.patch_merge = patch_merge

        # init
        print("============= init network =================")
        self.apply(self._init_weights)
        # self.intro.weight.data = torch.tensor(
        #                             [[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32
        #                         ).repeat(width, img_channel, 1, 1)

    def _init_weights(self, m: nn.Module):
        # print(type(m))
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def alter_ropes(self, ft_img_size):
        if ft_img_size != self.pt_img_size and self.rope:
            for rope in self.enc_ropes:
                rope.alter_seq_len(ft_img_size, rope.seq_len)
                ft_img_size //= 2

            self.middle_rope.alter_seq_len(ft_img_size, rope.seq_len)

            # for rope in self.middle_ropes:
            ft_img_size *= 2
            self.middle_rope.alter_seq_len(ft_img_size, rope.seq_len)

    def _forward_implem(self, inp, cond, c_shuffle=False):
        x = inp
        B, C, H, W = x.shape

        # x = torch.cat([x, cond], dim=1)
        x = self.intro(x)
        if self.if_abs_pos:
            x = x + self.abs_pos

        # x = self.square_relu(x)
        
        # naf_encs = []
        # for encoder, down in zip(self.naf_encoders, self.naf_downs):
        #     x = encoder.NAF_enc_forward(x, cond)
        #     naf_encs.append(x)
        #     x = down(x)

        lemm_encs = []
        encs_states = []
        states = [None, None]
        x = rearrange(x, 'b c h w -> b h w c')
        for i, (encoder, down) in enumerate(zip(self.lemm_encoders, self.lemm_downs)):
            # print(f'unisequenctial encoder {i}')
            # x = rope(x)
            # TODO: input previous state
            x, states = encoder.LEMM_enc_forward(x, cond, c_shuffle, states)
            lemm_encs.append(x)
            encs_states.append(states)
            x = down(x)

        # x = self.middle_rope(x)
        # print('unisequenctial middle layer')
        x, states = self.middle_blks.LEMM_enc_forward(x, cond, c_shuffle, states)

        for i, (decoder, up, skip_scale, enc_skip, enc_state_skip) in enumerate(zip(self.lemm_decoders, self.lemm_ups, self.skip_scales,
                                                                        lemm_encs[::-1], encs_states[::-1])):
            # print(f'unisequenctial decoder {i}')
            x = up(x)
            x = torch.cat([x, enc_skip * skip_scale], dim=-1)
            # print(x.shape[-1], enc_skip.shape[1])
            x, states = decoder.LEMM_dec_forward(x, cond, c_shuffle, states, enc_state_skip)
            
        x = rearrange(x, 'b h w c -> b c h w')
        # for decoder, up, enc_skip in zip(self.naf_decoders, self.naf_ups, naf_encs[::-1]):
        #     x = up(x)
        #     x = torch.cat([x, enc_skip], dim=1)
        #     x = decoder.NAF_dec_forward(x, cond)

        x = self.ending(x)
        
        return x

    @torch.no_grad()
    def val_step(self, ms, lms, pan, patch_merge=None):
        if patch_merge is None:
            patch_merge = self.patch_merge
            
        c_shuffle = False

        if patch_merge:
            _patch_merge_model = PatchMergeModule(
                self,
                crop_batch_size=8,
                patch_size_list=[16, 16 * self.upscale, 16 * self.upscale],
                scale=self.upscale,
                patch_merge_step=self.patch_merge_step,
            )
            sr = _patch_merge_model.forward_chop(ms, lms, pan, c_shuffle=c_shuffle)[0] + lms
        else:
            self.alter_ropes(pan.shape[-1])
            sr = self._forward_implem(lms, pan, c_shuffle) + lms
            self.alter_ropes(self.pt_img_size)

        return sr

    def train_step(self, ms, lms, pan, gt, criterion, c_shuffle=False):           
        sr = self._forward_implem(lms, pan, c_shuffle) + lms
        loss = criterion(sr, gt)

        return sr, loss

    def patch_merge_step(self, ms, lms, pan, **kwargs):
        sr = self._forward_implem(lms, pan, **kwargs)  # sr[:,[29,19,9]]
        return sr


    def to_dtype(self, dtype=torch.float32):
        for n, p in self.named_parameters():
            if 'op' in n:
                continue
            else: p.data = p.data.to(dtype)
        
        return self
    
    def print_dtype(self):
        for n, p in self.named_parameters():
            if 'op' in n:
                print('{:<80} {:>10}'.format(n, str(p.dtype)))
                