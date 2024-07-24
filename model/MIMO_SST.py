import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

import sys

sys.path.append("./")

from model.base_model import BaseModel, register_model, PatchMergeModule


##########################################################################
## Layer Norm


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Attention1(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention1, self).__init__()
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
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        return k, q, v


class Attention2(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention2, self).__init__()
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
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, q, k, v):
        b, c, h, w = x.shape

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class Attention3(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention3, self).__init__()
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
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, q, k, v):
        b, c, h, w = x.shape

        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        # Normalization (SoftMax)
        attn = F.softmax(attn, dim=-1)

        # Attention output
        output = torch.matmul(attn, v)

        # Reshape output to original format
        output = output.view(b, c, h, w)
        return output


class TransformerBlock1(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock1, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.attn1 = Attention1(dim, num_heads, bias)
        self.attn2 = Attention2(dim, num_heads, bias)
        self.attn3 = Attention3(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, xx):
        x, y = xx[0], xx[1]
        # x = x + self.attn(self.norm1(x))
        # x = x + self.ffn(self.norm2(x))
        x_k, x_q, x_v = self.attn1(self.norm1(x))

        # y = y + self.attn(self.norm1(y))
        # y = y + self.ffn(self.norm2(y))
        y_k, y_q, y_v = self.attn1(self.norm1(y))

        x = x + self.attn2(x, y_k, x_q, y_v)
        x = x + self.ffn(self.norm2(x))

        y = y + self.attn3(y, x_k, y_q, x_v)
        y = y + self.ffn(self.norm2(y))

        return x, y


class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
import math


class FeedForward(nn.Module):
    def __init__(
        self, dim, ffn_expansion_factor, hidden_dim=48, act_layer=nn.GELU, use_eca=False
    ):
        super().__init__()
        hidden_features = dim
        # self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
        #                            act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=3, stride=1, padding=1),
            act_layer(),
        )
        # self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        # self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(hidden_features) if use_eca else nn.Identity()
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(hidden_features, dim, kernel_size=3, stride=1, padding=1),
            act_layer(),
        )

    def forward(self, x):
        x = self.dwconv(x)
        x = self.eca(x)
        # x = self.dwconv1(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
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
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

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

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class Attention_spatio(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_spatio, self).__init__()
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
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        # Normalization (SoftMax)
        attn = F.softmax(attn, dim=-1)

        # Attention output
        output = torch.matmul(attn, v)

        # Reshape output to original format
        output = output.view(b, c, h, w)
        return output


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.attnspa = Attention_spatio(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)

        return x


# import PixelUnshuffle
##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            )
        )

    def forward(self, x):
        # modified here
        return torch.pixel_unshuffle(self.body(x), 2)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class BasicConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        bias=False,
        norm=False,
        relu=True,
        transpose=False,
        channel_shuffle_g=0,
        norm_method=nn.BatchNorm2d,
        groups=1,
    ):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                    groups=groups,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                    groups=groups,
                )
            )
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


import torch.fft
from torchvision.models import ResNet


class ResBlock_fft_bench(nn.Module):
    def __init__(self, n_feat, norm="backward"):  # 'ortho'
        super(ResBlock_fft_bench, self).__init__()
        self.main = nn.Sequential(
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=True),
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=False),
        )
        self.main_fft = nn.Sequential(
            BasicConv(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=True),
            BasicConv(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=False),
        )
        self.dim = n_feat
        self.norm = norm
        planes = n_feat
        self.planes = planes

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.fftn(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfftn(y, s=(H, W), norm=self.norm)

        # out = self.att(x)
        # return self.main(x)*out + x + y
        return self.main(x) + x


class ResBlock(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(
                out_channel, out_channel, kernel_size=3, stride=1, relu=True, norm=False
            ),
            BasicConv(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                relu=False,
                norm=False,
            ),
        )

    def forward(self, x):
        return self.main(x) + x


@register_model("MIMO_SST")
class Net(BaseModel):
    def __init__(self, in_chan=8):
        super(Net, self).__init__()
        pan_chan = 3
        inp_channels = 48*2+in_chan+pan_chan
        dim = 48
        num_blocks = [1, 1, 1, 1]
        heads = [1, 1, 1, 1]
        ffn_expansion_factor = 2.66
        
        bias = False
        LayerNorm_type = "WithBias"
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3

        self.down3_4 = Downsample(int(dim * 2**2))  ## From Level 3 to Level 4

        self.up4_3 = Upsample(int(dim * 2**3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )

        self.up2_1 = Upsample(
            int(dim * 2**1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        # self.upSample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.Conv96_31 = nn.Conv2d(
            in_channels=96, out_channels=in_chan, kernel_size=3, padding=(1, 1)
        )
        self.Conv192_31 = nn.Conv2d(
            in_channels=192, out_channels=in_chan, kernel_size=3, padding=(1, 1)
        )
        self.re = nn.ReLU(inplace=True).cuda()
        self.Conv144_96 = nn.Conv2d(
            in_channels=144, out_channels=96, kernel_size=3, padding=(1, 1)
        )
        self.Conv240_192 = nn.Conv2d(
            in_channels=240, out_channels=192, kernel_size=3, padding=(1, 1)
        )
        self.Conv1 = nn.Conv2d(
            in_channels=240, out_channels=192, kernel_size=3, padding=(1, 1)
        )
        self.encoder_level11 = nn.Sequential(
            *[
                TransformerBlock1(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.Conv3_64 = nn.Conv2d(
            in_channels=pan_chan, out_channels=48, kernel_size=3, padding=(1, 1)
        )
        self.Conv31_64 = nn.Conv2d(
            in_channels=in_chan, out_channels=48, kernel_size=3, padding=(1, 1)
        )

    def _forward_implem(self, x, y):
        # x: pan
        # y: lms

        # y = self.upSample(y)
        X = self.Conv3_64(x)
        Y = self.Conv31_64(y)
        # X = self.encoder_level1(X)
        # Y = self.encoder_level1(Y)
        XX1 = {0: X, 1: Y}
        X, Y = self.encoder_level11(XX1)
        Z = torch.cat([X, Y, x, y], dim=1)

        # 下采样2倍
        x1 = F.interpolate(x, scale_factor=0.5)
        y1 = F.interpolate(y, scale_factor=0.5)
        X1 = self.Conv3_64(x1)
        Y1 = self.Conv31_64(y1)

        XX2 = {0: X1, 1: Y1}
        X1, Y1 = self.encoder_level11(XX2)
        Z1 = torch.cat([X1, Y1, x1, y1], dim=1)
        inp_enc_level11 = self.patch_embed(Z1)

        # 下采样4倍
        x2 = F.interpolate(x1, scale_factor=0.5)
        y2 = F.interpolate(y1, scale_factor=0.5)
        X2 = self.Conv3_64(x2)
        Y2 = self.Conv31_64(y2)
        # X2 = self.encoder_level1(X2)
        # Y2 = self.encoder_level1(Y2)
        XX3 = {0: X2, 1: Y2}
        X2, Y2 = self.encoder_level11(XX3)
        Z2 = torch.cat([X2, Y2, x2, y2], dim=1)
        inp_enc_level12 = self.patch_embed(Z2)

        inp_enc_level1 = self.patch_embed(Z)

        inp_enc_level2 = self.down1_2(inp_enc_level1)
        ZZ1 = torch.cat([inp_enc_level2, inp_enc_level11], dim=1)
        ZZ1 = self.Conv144_96(ZZ1)

        inp_enc_level3 = self.down2_3(ZZ1)
        ZZ2 = torch.cat([inp_enc_level3, inp_enc_level12], dim=1)
        ZZ2 = self.Conv240_192(ZZ2)

        X1 = self.Conv192_31(ZZ2)
        X1 = y2 + X1

        out_dec_level3 = ZZ2
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, ZZ1], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        X2 = self.Conv96_31(inp_dec_level2)
        X2 = y1 + X2
        inp_dec_level1 = self.up2_1(inp_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, inp_enc_level1], 1)

        X = self.Conv96_31(inp_dec_level1)
        X = X + y
        X = self.re(X)

        if self.training:
            return X1, X2, X
        else:
            return X

    def train_step(self, ms, lms, pan, gt, criterion):
        sr_d4, sr_d2, sr = self._forward_implem(pan, lms)
        l1, loss_d4 = criterion(sr_d4, F.interpolate(gt, size=sr_d4.shape[-2:]))
        l2, loss_d2 = criterion(sr_d2, F.interpolate(gt, size=sr_d2.shape[-2:]))
        l3, loss_d1 = criterion(sr, gt)

        keys = list(loss_d1.keys())
        for k in keys:
            loss_d1[k] = (loss_d1[k] + loss_d2[k] + loss_d4[k]) / 3

        return sr, ((l1 + l2 + l3) / 3, loss_d1)

    @torch.no_grad()
    def val_step(self, ms, lms, pan, patch_merge=False):
        self.up_factor = 4
        if patch_merge:
            _patch_merge_model = PatchMergeModule(
                self,
                crop_batch_size=64,
                patch_size_list=[16, 16 * self.up_factor, 16 * self.up_factor],
                scale=self.up_factor,
                patch_merge_step=self.patch_merge_step,
            )
            sr = _patch_merge_model.forward_chop(ms, lms, pan)[0]
        else:
            sr = self._forward_implem(pan, lms)

        return sr

    def patch_merge_step(self, ms, lms, pan):
        sr = self._forward_implem(pan, lms)

        return sr


if __name__ == "__main__":
    from utils import get_loss

    torch.cuda.set_device("cuda:0")

    loss_fn = get_loss("l1ssim")

    ms = torch.randn(1, 31, 8, 8).cuda()
    lms = torch.randn(1, 31, 64, 64).cuda()
    pan = torch.randn(1, 3, 64, 64).cuda()
    gt = torch.randn(1, 31, 64, 64).cuda()

    net = Net(lms.shape[1]).cuda()
    # sr, loss = net.train_step(None, lms, pan, gt, loss_fn)
    # loss[0].backward()
    # print(sr.shape, loss)
    
    # net.eval()
    # sr = net.val_step(ms, lms, pan)
    
    # print(sr.shape)
    
    from fvcore.nn import flop_count_table, FlopCountAnalysis, parameter_count_table

    net.forward = net._forward_implem
    flops = FlopCountAnalysis(net, (pan, lms))
    print(flop_count_table(flops, max_depth=3))

    
    
