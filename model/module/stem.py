import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple


class GaussianHighPassModule(nn.Module):
    def __init__(self, in_c, h, w):
        super(GaussianHighPassModule, self).__init__()
        self.mx = nn.Parameter(torch.zeros(in_c, 1, 1))
        self.my = nn.Parameter(torch.zeros(in_c, 1, 1))
        self.sx = nn.Parameter(torch.randn(in_c, 1, 1))
        self.sy = nn.Parameter(torch.randn(in_c, 1, 1))
        self.in_c = in_c

        x = torch.linspace(-1, 1, h)
        y = torch.linspace(-1, 1, w)
        x, y = torch.meshgrid(x, y, indexing='ij')  # get 2D variables instead of 1D
        x = x[None, ...].repeat(in_c, 1, 1)
        y = y[None, ...].repeat(in_c, 1, 1)
        self.register_buffer('x_buffer', x)
        self.register_buffer('y_buffer', y)

    def gaus2d(self, x, y, mx, my, sx, sy):
        # x, y shape is [c, h, w]
        return 1. / (2. * torch.pi * sx * sy) * torch.exp(
            -((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))

    def forward(self, x):
        z = self.gaus2d(self.x_buffer, self.y_buffer, self.mx, self.my, self.sx, self.sy)  # [c, h, w]
        freq_x = torch.fft.fft2(x)
        freq_x = torch.fft.fftshift(freq_x)
        freq_x = freq_x * z.unsqueeze(0)
        freq_x = F.normalize(freq_x, dim=1)

        return torch.fft.ifft2(torch.fft.ifftshift(freq_x)).abs()


class ViTStem(nn.Module):
    def __init__(self, in_c=3, out_c=64, patch_size=(4, 4), norm_lay=None):
        super(ViTStem, self).__init__()
        self.patch_size = patch_size
        # p = patch_size[0]

        self.stem_conv = nn.Conv2d(in_c, out_c, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_lay(out_c) if norm_lay is not None else None

    def forward(self, x):
        # B, C, H, W = x.shape
        # assert H == self.h and W == self.w, 'wrong size'

        x = self.stem_conv(x).flatten(2).transpose(1, 2)  # B ph*pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

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

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


if __name__ == '__main__':
    ghpm = GaussianHighPassModule(3, 64, 64)
    x = torch.randn(1, 3, 64, 64)
    x2 = ghpm(x)
    x2.sum().backward()
    print(ghpm.sx.grad)
