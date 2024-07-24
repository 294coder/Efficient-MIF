import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class Resblock(nn.Module):
    def __init__(self, channel=32, ksize=3, padding=1):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=ksize,
                               padding=padding,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=ksize,
                               padding=padding,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv1(x))  # Bsx32x64x64
        rs1 = self.conv2(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


class Convnxn(nn.Module):
    scales = {1: (1, 1, 0), 3: (3, 1, 1), 5: (5, 1, 2), 7: (7, 1, 3)}

    def __init__(self, in_c, out_c, scale):
        super(Convnxn, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, *self.scales[scale])
        self.conv2 = nn.Conv2d(out_c, out_c, *self.scales[scale])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, channel, hidden_c):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(channel, hidden_c, 1, 1, 0)
        self.conv2 = nn.Conv2d(hidden_c, hidden_c, 3, 1, 1)
        self.conv3 = nn.Conv2d(hidden_c, channel, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return self.relu(x + short_cut)


class ConvXBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, ksize, padding, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=ksize, padding=padding, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class DWConv(nn.Module):
    def __init__(self, embed_dim=768, kernel_size=3, padding=2):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(embed_dim, embed_dim, kernel_size, stride=1, padding=padding, bias=True,
                                groups=embed_dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class MultiScaleEncoder(nn.Module):
    def __init__(self, in_c, hidden_c=(16, 32, 64), out_c=32, ksize=(3, 5, 7), padding=(1, 2, 3)):
        super().__init__()

        # TODO: TO keep size unchanged, padding = ksize // 2
        self.in_c = in_c
        self.conv_stage1 = nn.Sequential(
            nn.Conv2d(in_c + 1, hidden_c[0], 1),
            # ConvXBlock(hidden_c[0], ksize[0], padding[0]),
            # ConvXBlock(hidden_c[0], ksize[0], padding[0])
            Resblock(hidden_c[0], ksize[0], padding[0]),
            Resblock(hidden_c[0], ksize[0], padding[0]),
        )
        self.conv_stage2 = nn.Sequential(
            nn.Conv2d(hidden_c[0], hidden_c[1], 1, 1, 0),
            ConvXBlock(hidden_c[1], ksize[1], padding[1]),
            ConvXBlock(hidden_c[1], ksize[1], padding[1]),
            # Resblock(hidden_c[1], ksize[1], padding[1]),
            # Resblock(hidden_c[1], ksize[1], padding[1]),
            nn.Conv2d(hidden_c[1], hidden_c[1] // 4, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2)
        )
        self.conv_stage3 = nn.Sequential(
            nn.Conv2d(hidden_c[1], hidden_c[2], 1, 1, 0),
            ConvXBlock(hidden_c[2], ksize[2], padding[2]),
            ConvXBlock(hidden_c[2], ksize[2], padding[2]),
            # Resblock(hidden_c[2], ksize[2], padding[2]),
            # Resblock(hidden_c[2], ksize[2], padding[2]),
            nn.Conv2d(hidden_c[2], hidden_c[2] // 4, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, lms, pan):
        # lms = F.interpolate(ms, scale_factor=4)
        # pan_concat = pan.repeat(1, self.in_c, 1, 1)
        # x = pan_concat - lms

        x = torch.cat([lms, pan], dim=1)

        x1 = self.conv_stage1(x)  # [64, 64]
        x2 = self.conv_stage2(x1)  # [32, 32]
        x3 = self.conv_stage3(x2)  # [16, 16]

        # p1 = self.down12(pan)
        # p2 = self.down23(pan)
        # p3 = self.down34(pan)
        #
        # r1 = p1 - x1
        # r2 = p2 - x2
        # r3 = p3 - x3

        return x1, x2, x3


if __name__ == '__main__':
    net = MultiScaleEncoder(8)
    x = torch.randn(1, 8, 64, 64)
    x2 = torch.randn(1, 1, 64, 64)
    y1, y2, y3 = net(x, x2)
    print(y1.shape, y2.shape, y3.shape)
