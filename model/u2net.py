import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y=None, **kwargs):
        if y is not None:
            return self.fn(x, y, **kwargs) + x
        else:
            return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y=None, **kwargs):
        if y is not None:
            return self.fn(self.norm(x), self.norm(y), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        # self.temperature = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.sa1 = nn.Linear(dim, inner_dim, bias=False)
        self.sa2 = nn.Linear(dim, inner_dim, bias=False)
        self.se1 = nn.Linear(dim, inner_dim, bias=False)
        self.se2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, y, mask=None):
        b, n, _, h = *x.shape, self.heads
        y1 = rearrange(self.sa1(y), 'b n (h d) -> b h n d', h=h)
        y2 = rearrange(self.sa2(y), 'b n (h d) -> b h n d', h=h)
        x1 = rearrange(self.se1(x), 'b n (h d) -> b h n d', h=h)
        x2 = rearrange(self.se2(x), 'b n (h d) -> b h n d', h=h)
        sacm = (y1 @ y2.transpose(-2, -1)) * self.scale
        secm = (x1.transpose(-2, -1) @ x2) * self.scale / (n/self.dim_head)  # b h d d
        sacm = sacm.softmax(dim=-1)
        secm = secm.softmax(dim=-1)
        out1 = torch.einsum('b h i j, b h j d -> b h i d', sacm, x1)
        out2 = torch.einsum('b h n i, b h i j -> b h n j', y1, secm)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out = out1 * out2
        out = self.to_out(out)
        return out


class S2Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, depth=1, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, MLP(dim, hidden_dim=mlp_dim, dropout=dropout)))]))

    def forward(self, x, y, mask=None):
        H = x.shape[2]
        x = rearrange(x, 'B C H W -> B (H W) C', H=H)
        y = rearrange(y, 'B C H W -> B (H W) C', H=H)
        for attn, ff in self.layers:
            x = attn(x, y, mask=mask)
            x = ff(x)
        x = rearrange(x, 'B (H W) C -> B C H W', H=H)
        return x
    
#### U2net Main ####
import torch
import torch.nn as nn
# from model.s2block import S2Block


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.conv1 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        rs1 = self.relu(self.conv0(x))
        rs1 = self.conv1(rs1)
        rs = torch.add(x, rs1)
        return rs


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.upsamle = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*16, 3, 1, 1, bias=False),
            nn.PixelShuffle(4)
        )

    def forward(self, x):
        return self.upsamle(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        else:
            self.up0 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.LeakyReLU()
            )
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0),
                nn.LeakyReLU()
            )
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        x1 = self.up1(x1)
        x = x1 + x2
        return self.relu(self.conv(x))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 2, 2, 0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class U2Net(nn.Module):
    def __init__(self, ms_dim, pan_dim, dim, dim_head=16, se_ratio_mlp=0.5, se_ratio_rb=0.5):
        super().__init__()
        # ms_dim = 8
        # pan_dim = 1

        self.relu = nn.LeakyReLU()
        self.upsample = Upsample(ms_dim)
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(ms_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(pan_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.to_hrms = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, ms_dim, 3, 1, 1)
        )
        
        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)
        dim3 = dim1
        dim4 = dim0

        # layer 0
        self.s2block0 = S2Block(dim0, dim0//dim_head, dim_head, int(dim0*se_ratio_mlp))
        self.down0 = Down(dim0, dim1)
        self.resblock0 = ResBlock(dim0, int(se_ratio_rb*dim0), dim0)

        # layer 1
        self.s2block1 = S2Block(dim1, dim1//dim_head, dim_head, int(dim1*se_ratio_mlp))
        self.down1 = Down(dim1, dim2)
        self.resblock1 = ResBlock(dim1, int(se_ratio_rb*dim1), dim1)

        # layer 2
        self.s2block2 = S2Block(dim2, dim2//dim_head, dim_head, int(dim2*se_ratio_mlp))
        self.up0 = Up(dim2, dim3)
        self.resblock2 = ResBlock(dim2, int(se_ratio_rb*dim2), dim2)

        # layer 3
        self.s2block3 = S2Block(dim3, dim3//dim_head, dim_head, int(dim3*se_ratio_mlp))
        self.up1 = Up(dim3, dim4)
        self.resblock3 = ResBlock(dim3, int(se_ratio_rb*dim3), dim3)

        # layer 4
        self.s2block4 = S2Block(dim4, dim4//dim_head, dim_head, int(dim4*se_ratio_mlp))

    def forward(self, x, y):
        x = self.upsample(x)
        skip_c0 = x
        x = self.raise_ms_dim(x)
        y = self.raise_pan_dim(y)

        # layer 0
        x = self.s2block0(x, y)  # 32 64 64
        skip_c10 = x  # 32 64 64
        x = self.down0(x)  # 64 32 32
        y = self.resblock0(y)  # 32 64 64
        skip_c11 = y  # 32 64 64
        y = self.down0(y)  # 64 32 32

        # layer 1
        x = self.s2block1(x, y)  # 64 32 32
        skip_c20 = x
        x = self.down1(x)  # 128 16 16
        y = self.resblock1(y)  # 64 32 32
        skip_c21 = y  # 64 32 32
        y = self.down1(y)  # 128 16 16

        # layer 2
        x = self.s2block2(x, y)  # 128 16 16
        x = self.up0(x, skip_c20)  # 64 32 32
        y = self.resblock2(y)  # 128 16 16
        y = self.up0(y, skip_c21)  # 64 32 32

        # layer 3
        x = self.s2block3(x, y)  # 64 32 32
        x = self.up1(x, skip_c10)  # 32 64 64
        y = self.resblock3(y)  # 64 32 32
        y = self.up1(y, skip_c11)  # 32 64 64

        # layer 4
        x = self.s2block4(x, y)  # 32 64 64
        output = self.to_hrms(x) + skip_c0  # 8 64 64
        
        return output


def summaries(model, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 16, 16), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)


if __name__ == '__main__':
    import time
    
    dim_ms, dim_pan = 31, 3
    img_size = 128
    scale = 4
    device = 'cuda:0'
    
    ms = torch.randn(1, dim_ms, img_size, img_size).to(device)
    pan = torch.randn(1, dim_pan, img_size*scale, img_size*scale).to(device)
    
    net = U2Net(dim_ms, dim_pan, 32).to(device)
    net.eval()
    
    from model.base_model import PatchMergeModule
    net = PatchMergeModule(net, crop_batch_size=32, patch_size_list=[16, 64], scale=scale, device=device,
                           patch_merge_step=net.forward)
    
    t1 = time.time()
    with torch.no_grad():
        for _ in range(20):
            out = net.forward_chop(ms, pan)
            
    t2 = time.time()
    print('time: ', (t2-t1)/20)
    
    
    # from fvcore.nn import flop_count_table, FlopCountAnalysis
    
    # print(flop_count_table(
    #     FlopCountAnalysis(net, (ms, pan))
    # ))
    