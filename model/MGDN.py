import os
import sys
import time
import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
from einops import rearrange
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# from pympler import tracker, classtracker

import sys
sys.path.append('./')
from model.base_model import PatchMergeModule, BaseModel, register_model

from utils import easy_logger

logger = easy_logger()


# from pympler import tracker
# tr = tracker.SummaryTracker()


### taken from swinir

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
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
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

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


class LocalContextExtractor(nn.Module):

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        B, C, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class ContextAwareTransformer(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
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
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.lce = LocalContextExtractor(self.dim)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # local context features
        lcf = x.permute(0, 3, 1, 2)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # local context
        lc = self.lce(lcf)
        lc = lc.view(B, C, H * W).permute(0, 2, 1)
        x = lc + x

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            ContextAwareTransformer(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size) # B L C
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class ContextAwareTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.dilated_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=2, bias=True, dilation=2)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.dilated_conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, kernel_size=3, padding=2, dilation=2), 
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, kernel_size=3, padding=2, dilation=2)
                )

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        res = self.residual_group(x, x_size) # B L C
        res = self.patch_unembed(res, x_size) # B c H W
        res = self.dilated_conv(res)
        res = self.patch_embed(res) + x
        return res

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B C H W ==> B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class SpatialAttentionModule(nn.Module):

    def __init__(self, dim):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map


class HDRTransformer_backbone(nn.Module):

    def __init__(self, img_size=128, patch_size=1, in_chans=64,
                 embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(HDRTransformer_backbone, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        
        ################################### 2. HDR Reconstruction Network ###################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Context-aware Transformer Blocks (CTBs)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ContextAwareTransformerBlock(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        # x [B, embed_dim, h, w]
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x) # B L C
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        res = self.conv_after_body(self.forward_features(x) + x)
        return res

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        return flops

#### some helper layers

def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True, act='LeakyReLU'):
    if act is not None:
        if act == 'LeakyReLU':
            act_ = nn.LeakyReLU(0.1,inplace=True)
        elif act == 'Sigmoid':
            act_ = nn.Sigmoid()
        elif act == 'Tanh':
            act_ = nn.Tanh()

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
            act_
        )
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias)
    
class CrossTransAttention(nn.Module):
    def __init__(self,num_heads,dim):
        super(CrossTransAttention, self).__init__()
        self.num_heads = num_heads
        bias=True
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim*1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, feat_guide,feat_op):
        # feat_ref: Guidance
        # feat_ext: Value
        b,c,h,w = feat_guide.shape
        
        q = self.q_dwconv(self.q(feat_guide))
        kv = self.kv_dwconv(self.kv(feat_op))
        k,v = kv.chunk(2, dim=1)
        
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

class DeepMuGIF(nn.Module):
    def __init__(self,GF_chans):
        super(DeepMuGIF, self).__init__()
        
        self.kernel_width = 7
        self.channel = GF_chans
        self.kernel_dim = self.kernel_width*self.kernel_width
        self.pks = 3
        
        self.TCA = CrossTransAttention(num_heads=6,dim=self.channel)
        
        self.kernel_predictor_ref = nn.Sequential(
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.kernel_dim, kernel_size=1, stride=1,act=None)
        )
        
        self.kernel_predictor_ext = nn.Sequential(
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.kernel_dim, kernel_size=1, stride=1,act=None)
        )
        
        self.conv_out_1 = conv(self.channel, self.channel, kernel_size=1, stride=1)
        self.conv_out_2 = conv(self.channel, self.channel, kernel_size=1, stride=1)

    def FAC(self, feat_in, kernel, ksize):
        """
        customized FAC
        """
        channels = feat_in.size(1)
        N, kernels, H, W = kernel.size()
        pad = (ksize - 1) // 2

        feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
        feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
        feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
        feat_in = feat_in.reshape(N, H, W, channels, -1)

        if channels ==3 and kernels == ksize*ksize:
            ####
            kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, 1, ksize, ksize)
            kernel = torch.cat([kernel,kernel,kernel],channels)
            kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1) 

        else:
            kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, 1, ksize, ksize) 
            kernel = kernel.repeat(1,1,1, self.channel,1,1)
            kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1) 

        feat_out = torch.sum(feat_in * kernel, -1)
        feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
        return feat_out

    def kernelpredict_ref(self, feat_ref, feat_ext):
        feat_kernel = self.TCA(feat_guide=feat_ref,feat_op=feat_ext)
        pre_kernel = self.kernel_predictor_ref(feat_kernel)
        return pre_kernel
    
    def kernelpredict_ext(self, feat_ref, feat_ext):
        feat_kernel = self.TCA(feat_guide=feat_ref,feat_op=feat_ext)
        pre_kernel = self.kernel_predictor_ext(feat_kernel)
        return pre_kernel

    def forward(self,feat_1,feat_2):
        # feat_1,feat_2 = feat_list
        kernel_1 = self.kernelpredict_ref(feat_ref=feat_1,feat_ext=feat_2)
        kernel_2 = self.kernelpredict_ext(feat_ref=feat_2,feat_ext=feat_1)
        
        out_feat_1 = self.FAC(feat_1, kernel_1, self.kernel_width)
        out_feat_2 = self.FAC(feat_2, kernel_2, self.kernel_width)
        
        out_feat_1 = self.conv_out_1(out_feat_1)
        out_feat_2 = self.conv_out_2(out_feat_2)
        
        return [out_feat_1,out_feat_2]
    
class HDR_DeepMuGIF(nn.Module):
    def __init__(self,GF_chans=30):
        super(HDR_DeepMuGIF, self).__init__()
        print("HDR_DeepMuGIF in model base")
        self.kernel_width = 7
        self.channel = GF_chans
        self.kernel_dim = self.kernel_width*self.kernel_width
        self.pks = 3
        
        self.TCA = CrossTransAttention(num_heads=6,dim=self.channel)
                
        self.kernel_predictor_under = nn.Sequential(
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.kernel_dim, kernel_size=1, stride=1,act=None)
        )
        
        self.kernel_predictor_over = nn.Sequential(
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.kernel_dim, kernel_size=1, stride=1,act=None)
        )
        
        self.conv_out_1 = conv(self.channel, self.channel, kernel_size=1, stride=1)
        self.conv_out_2 = conv(self.channel, self.channel, kernel_size=1, stride=1)
        
        self.conv_ref = nn.Sequential(
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
        )

    def FAC(self, feat_in, kernel, ksize):
        """
        customized FAC
        """
        channels = feat_in.size(1)
        N, kernels, H, W = kernel.size()
        pad = (ksize - 1) // 2

        feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
        feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
        feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
        feat_in = feat_in.reshape(N, H, W, channels, -1)

        if channels ==3 and kernels == ksize*ksize:
            ####
            kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, 1, ksize, ksize)
            kernel = torch.cat([kernel,kernel,kernel],channels)
            kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1) 

        else:
            kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, 1, ksize, ksize) 
            kernel = kernel.repeat(1,1,1, self.channel,1,1)
            kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1) 

        feat_out = torch.sum(feat_in * kernel, -1)
        feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
        return feat_out

    def kernelpredict_under(self, feat_ref, feat_ext):
        feat_kernel = self.TCA(feat_guide=feat_ref,feat_op=feat_ext)
        pre_kernel = self.kernel_predictor_under(feat_kernel)
        return pre_kernel
    
    def kernelpredict_over(self, feat_ref, feat_ext):
        feat_kernel = self.TCA(feat_guide=feat_ref,feat_op=feat_ext)
        pre_kernel = self.kernel_predictor_over(feat_kernel)
        return pre_kernel


    def forward(self,feat_1,feat_2,feat_3):
        # feat_1,feat_2,feat_3 = feat_list
        
        kernel_1 = self.kernelpredict_under(feat_ref=feat_2,feat_ext=feat_1)
        kernel_3 = self.kernelpredict_over(feat_ref=feat_2,feat_ext=feat_3)
        
        out_feat_1 = self.FAC(feat_1, kernel_1, self.kernel_width)
        out_feat_3 = self.FAC(feat_3, kernel_3, self.kernel_width)
        
        out_feat_1 = self.conv_out_1(out_feat_1)
        out_feat_3 = self.conv_out_2(out_feat_3)
        
        out_feat_2 = self.conv_ref(feat_2)
        
        return [out_feat_1,out_feat_2,out_feat_3]
    

############################################# Main Net ######################################################

class MGFF_Blocks(nn.Module):

    def __init__(self, GF_chans, input_dim, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 kernel_num=4, inference=False, mid_channel=16, temperature=34,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.MGF = DeepMuGIF(GF_chans)
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.dilated_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=2, bias=True, dilation=2)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.dilated_conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, kernel_size=3, padding=2, dilation=2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, kernel_size=3, padding=2, dilation=2)
            )

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
        #     norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_embed_MGF = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.adjust_channel = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_01 = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_02 = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_03 = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0, bias=True)

    def forward(self, feat_list):
        # Mutual Guided Fliter
        outlist = self.MGF(feat_list[0], feat_list[1])
        x = torch.cat((outlist[0], outlist[1]), dim=1)
        x = self.adjust_channel(x)
        # Fusion
        # x [B, embed_dim, h, w]
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed_MGF(x)  # B L C
        res = self.residual_group(x, x_size)  # B L C
        res = self.patch_unembed(res, x_size)  # B c H W
        res = self.dilated_conv(res)
        # concat input
        out_out_1 = self.adjust_channel_01(torch.cat((res, outlist[0]), dim=1))
        out_out_2 = self.adjust_channel_02(torch.cat((res, outlist[1]), dim=1))
        return [out_out_1, out_out_2]

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        flops += self.patch_embed_MGF.flops()
        return flops
    
class MGFF_HDR_Blocks(nn.Module):

    def __init__(self, GF_chans, input_dim, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 kernel_num=4,inference=False,mid_channel=16,temperature=34,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.MGF = HDR_DeepMuGIF(GF_chans=60)
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.dilated_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=2, bias=True, dilation=2)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.dilated_conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, kernel_size=3, padding=2, dilation=2), 
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, kernel_size=3, padding=2, dilation=2)
                )

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
        #     norm_layer=None)
        
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_embed_MGF = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)
        
        self.adjust_channel = nn.Conv2d(3*dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_01 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_02 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_03 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0, bias=True)

    def forward(self, feat_list):
        # Mutual Guided Fliter
        outlist = self.MGF(feat_list[0],feat_list[1],feat_list[2])
        x = torch.cat((outlist[0],outlist[1],outlist[2]),dim=1)
        x = self.adjust_channel(x)
        # Fusion
        # x [B, embed_dim, h, w]
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed_MGF(x) # B L C
        res = self.residual_group(x, x_size) # B L C
        res = self.patch_unembed(res, x_size) # B c H W
        res = self.dilated_conv(res)
        # concat input
        out_out_1 =  self.adjust_channel_01(torch.cat((res,outlist[0]),dim=1))
        out_out_2 =  self.adjust_channel_02(torch.cat((res,outlist[1]),dim=1))
        out_out_3 =  self.adjust_channel_03(torch.cat((res,outlist[2]),dim=1))
        return [out_out_1,out_out_2,out_out_3]

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        flops += self.patch_embed_MGF.flops()
        return flops
    

@register_model('MGDN')
class MGFF(BaseModel):
    def __init__(self, GF_chans=30, img_size=128, patch_size=1, in_chans=3,
                 embed_dim=48, depths=[3, 3], num_heads=[6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 kernel_num=4, inference=False, mid_channel=16, temperature=34,
                 use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(MGFF, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans

        ################################### 2. HDR Reconstruction Network ###################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.GF_chans = GF_chans

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.inconv1 = nn.Conv2d(3, embed_dim, kernel_size=3, padding=1, bias=True)
        self.inconv2 = nn.Conv2d(1, embed_dim, kernel_size=3, padding=1, bias=True)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Context-aware Transformer Blocks (CTBs)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MGFF_Blocks(
                GF_chans=self.GF_chans,
                input_dim=embed_dim,
                dim=embed_dim,
                input_resolution=(patches_resolution[0],
                                  patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                kernel_num=kernel_num[i_layer],
                inference=inference,
                mid_channel=mid_channel,
                temperature=temperature,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        self.conv_after_body = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(embed_dim, 3, 3, 1, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def update_temperature(self):
    #     for m in self.modules():
    #         if isinstance(m, MuGIF):
    #             m.update_temperature()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x[0].shape[2], x[0].shape[3])
        for layer in self.layers:
            x = layer(x)
        return x

    def _forward_implem(self, under, over):

        _, _, H, W = under.shape

        under = self.check_image_size(under)
        over = self.check_image_size(over)

        under_feat = self.inconv1(under)
        over_feat = self.inconv2(over)

        x_list = [under_feat, over_feat]

        feat = torch.cat(self.forward_features(x_list), dim=1)

        res = self.conv_after_body(feat) + under_feat

        output = self.conv_last(res)

        return output[:, :, :H, :W]


    def fusion_train_step(self, vis, ir, mask, gt, criterion):
        fused = self._forward_implem(vis, ir)
        loss = criterion(fused, gt)
        
        return fused, loss
    
    @torch.no_grad()
    def fusion_val_step(self, vis, ir, mask, *, patch_merge=False):
        try:
            if not patch_merge:
                fused = self._forward_implem(vis, ir)
            else:
                pm_fn = PatchMergeModule(patch_merge_step=self._forward_implem,
                                patch_size_list=[64, 64],
                                scale=1,
                                crop_batch_size=16)
                fused = pm_fn.forward_chop(vis, ir)[0]
        except Exception:
            logger.warning('GPU OOM! try to use PatchMergeModule to forward every patch')
            pm_fn = PatchMergeModule(patch_merge_step=self._forward_implem,
                                patch_size_list=[64, 64],
                                scale=1,
                                crop_batch_size=16)
            fused = pm_fn.forward_chop(vis, ir)[0]
        
        return fused

    def check_image_size(self, x):
        # print("[debug1,check_image_size]==>",x.shape)
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # print("[debug2,check_image_size]==>",x.shape)
        return x

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        return flops


class MGFF_GDSR(nn.Module):
    def __init__(self, GF_chans=30, img_size=128, patch_size=1, in_chans=3,
                embed_dim=48, depths=[3, 3], num_heads=[6, 6],
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                kernel_num=4,inference=False,mid_channel=16,temperature=34,
                use_checkpoint=False, resi_connection='1conv',
                **kwargs):
        super(MGFF_GDSR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        
        ################################### 2. HDR Reconstruction Network ###################################
        self.num_layers = len(depths)
        
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # split image into non-depthlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-depthlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.inconv1 = nn.Conv2d(3, embed_dim, kernel_size=3, padding=1, bias=True)
        self.inconv2 = nn.Conv2d(1, embed_dim, kernel_size=3, padding=1, bias=True)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Context-aware Transformer Blocks (CTBs)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MGFF_Blocks(
                        GF_chans = GF_chans,
                        input_dim=embed_dim,
                        dim=embed_dim,
                        input_resolution=(patches_resolution[0],
                                        patches_resolution[1]),
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                        norm_layer=norm_layer,
                        downsample=None,
                        use_checkpoint=use_checkpoint,
                        img_size=img_size,
                        patch_size=patch_size,
                        resi_connection=resi_connection,
                        kernel_num=kernel_num[i_layer],
                        inference=inference,
                        mid_channel=mid_channel,
                        temperature=temperature,
                        )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        
        self.conv_after_body = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(embed_dim, 1, 3, 1, 1),
            nn.Sigmoid()
            )          

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    # def update_temperature(self):
    #     for m in self.modules():
    #         if isinstance(m, MuGIF):
    #             m.update_temperature()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x[0].shape[2], x[0].shape[3])
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, rgb, depth):
        
        _,_,H,W = rgb.shape
        
        rgb = self.check_image_size(rgb)
        depth = self.check_image_size(depth)
        
        rgb_feat = self.inconv1(rgb)
        depth_feat = self.inconv2(depth)
        
        x_list = [rgb_feat,depth_feat]
        
        feat = torch.cat(self.forward_features(x_list),dim=1)
        
        res = self.conv_after_body(feat) + depth_feat
        
        output = self.conv_last(res)
        
        return output[:,:,:H,:W]
    
    def check_image_size(self, x):
        #print("[debug1,check_image_size]==>",x.shape)
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        #print("[debug2,check_image_size]==>",x.shape)
        return x

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        return flops


class MGFF_Deghosting(nn.Module):
    def __init__(self, img_size=128, patch_size=1, in_chans=3,
                 embed_dim=48, depths=[3, 3], num_heads=[6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 kernel_num=4,inference=False,mid_channel=16,temperature=34,
                 use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(MGFF_Deghosting, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        
        ################################### 2. HDR Reconstruction Network ###################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.conv1 = nn.Conv2d(6, embed_dim, kernel_size=3, padding=1, bias=True)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Context-aware Transformer Blocks (CTBs)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MGFF_HDR_Blocks(
                         GF_chans=60,
                         input_dim=embed_dim,
                         dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection,
                         kernel_num=kernel_num[i_layer],
                         inference=inference,
                         mid_channel=mid_channel,
                         temperature=temperature,
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        
        self.conv_after_body = nn.Sequential(
            nn.Conv2d(embed_dim*3, embed_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(embed_dim, 3, 3, 1, 1),
            nn.Sigmoid()
            )          

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    # def update_temperature(self):
    #     for m in self.modules():
    #         if isinstance(m, Odconv_Dynamic_Encoder):
    #             m.update_temperature()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x[0].shape[2], x[0].shape[3])
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, under, ref, over):
        
        _,_,H,W = ref.shape
        
        under = self.check_image_size(under)
        ref = self.check_image_size(ref)
        over = self.check_image_size(over)
        
        under_feat = self.conv1(under)
        ref_feat = self.conv1(ref)
        over_feat = self.conv1(over)
        
        x_list = [under_feat,ref_feat,over_feat]
        
        feat = torch.cat(self.forward_features(x_list),dim=1)
        
        res = self.conv_after_body(feat)+ref_feat
        
        output = self.conv_last(res)
        
        # print("===>",output[:,:,:H,:W].shape)
        return output[:,:,:H,:W]
    
    def check_image_size(self, x):
        #print("[debug1,check_image_size]==>",x.shape)
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        #print("[debug2,check_image_size]==>",x.shape)
        return x

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        return flops


# tr = classtracker.ClassTracker()
# tr.track_class(MGFF)
# tr.create_snapshot()
# print('create mem tracker')


if __name__ == "__main__":
    import argparse
    from thop.profile import profile
    import time
    
    torch.cuda.set_device(0)

    
    # our         | 0.8773(M)      | 59.5266(G)         |
    model = MGFF(GF_chans=60, 
                 embed_dim=60,
                 depths=[6, 6],
                 num_heads=[6, 6],
                 window_size=8,
                 kernel_num=[1, 1, 1],
                 mlp_ratio=2.,
                 inference=False).cuda()
    height = 64
    width = 64

    while True:
        rgb = torch.randn((16, 3, height, width)).cuda()
        depth = torch.randn((16, 1, height, width)).cuda()
        
        ## test `fusion_val_step`
        with torch.no_grad():
            fused, _ = model.fusion_train_step(rgb, depth, mask=None, gt=None, criterion=lambda *args: None)
        print(fused.shape)
    
    
    ## cuda comsuption

    # print(
    #     torch.cuda.memory_summary(abbreviated=False)
    # )