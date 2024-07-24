import torch
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
import torch.nn.init

from model.base_model import BaseModel, register_model

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes//4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//4, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_qm = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_km = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_vm = nn.Linear(dim, dim_head * heads, bias=False)        
        self.to_k2 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v2 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k4 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v4 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k8 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v8 = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescalem = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale2 = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale4 = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale8 = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_2,x_4,x_8,y):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = y.shape
        x2 = x_2.reshape(b,h*w,c)
        x4 = x_4.reshape(b,h*w,c)
        x8 = x_8.reshape(b,h*w,c)
        y = y.reshape(b,h*w,c)

        q_inpm = self.to_qm(y)
        k_inpm = self.to_km(y)
        v_inpm = self.to_vm(y)
        qm, km, vm = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inpm, k_inpm, v_inpm))
        vm = vm
        # q: b,heads,hw,c
        qm = qm.transpose(-2, -1)
        km = km.transpose(-2, -1)
        vm = vm.transpose(-2, -1)
        qm = F.normalize(qm, dim=-1, p=2)
        km = F.normalize(km, dim=-1, p=2)
        attnm = (km @ qm.transpose(-2, -1))   # A = K^T*Q
        attnm = attnm * self.rescalem
        attnm = attnm.softmax(dim=-1)

        k_inp2 = self.to_k2(x2)
        v_inp2 = self.to_v2(x2)
        k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (k_inp2, v_inp2))
        v2 = v2
        # q: b,heads,hw,c
        k2 = k2.transpose(-2, -1)
        v2 = v2.transpose(-2, -1)
        k2 = F.normalize(k2, dim=-1, p=2)
        attn2 = (k2 @ qm.transpose(-2, -1))   # A = K^T*Q
        attn2 = attn2 * self.rescale2
        attn2 = attn2.softmax(dim=-1)

        k_inp4 = self.to_k4(x4)
        v_inp4 = self.to_v4(x4)
        k4, v4 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (k_inp4, v_inp4))
        v4 = v4
        # q: b,heads,hw,c
        k4 = k4.transpose(-2, -1)
        v4 = v4.transpose(-2, -1)
        k4 = F.normalize(k4, dim=-1, p=2)
        attn4 = (k4 @ qm.transpose(-2, -1))   # A = K^T*Q
        attn4 = attn4 * self.rescale4
        attn4 = attn4.softmax(dim=-1)

        k_inp8 = self.to_k8(x8)
        v_inp8 = self.to_v8(x8)
        k8, v8 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (k_inp8, v_inp8))
        v8 = v8
        # q: b,heads,hw,c
        k8 = k8.transpose(-2, -1)
        v8 = v8.transpose(-2, -1)
        k8 = F.normalize(k8, dim=-1, p=2)
        attn8 = (k8 @ qm.transpose(-2, -1))   # A = K^T*Q
        attn8 = attn8 * self.rescale4
        attn8 = attn8.softmax(dim=-1)

        x = attnm @ vm +  attn2 @ v2 + attn4 @ v4 + attn8 @ v8  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inpm.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class MLSIF(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x_2,x_4,x_8,y):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x_2 = x_2.permute(0, 2, 3, 1)
        x_4 = x_4.permute(0, 2, 3, 1)
        x_8 = x_8.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x_2,x_4,x_8,y) + y
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class SpePyBlock(nn.Module):
    def __init__(self, inchannels, bias=True):
        super(SpePyBlock, self).__init__()
        self.conv2 = nn.Sequential(
                        nn.Conv2d(inchannels, inchannels*2, kernel_size=3, stride=1, padding=1, groups=2, bias=bias),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(inchannels*2, inchannels*2, kernel_size=3, stride=1, padding=1, groups=2, bias=bias),
                        nn.LeakyReLU(0.2)
                        )

        self.conv4 = nn.Sequential(
                        nn.Conv2d(inchannels, inchannels*2, kernel_size=3, stride=1, padding=1, groups=4, bias=bias),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(inchannels*2, inchannels*2, kernel_size=3, stride=1, padding=1, groups=4, bias=bias),
                        nn.LeakyReLU(0.2)
                        )
        
        self.conv8 = nn.Sequential(
                        nn.Conv2d(inchannels, inchannels*2, kernel_size=3, stride=1, padding=1, groups=8, bias=bias),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(inchannels*2, inchannels*2, kernel_size=3, stride=1, padding=1, groups=8, bias=bias),
                        nn.LeakyReLU(0.2)
                        )


    def forward(self, x2,x4,x8):
        _, c, _, _ = x2.shape
        if c % 8 != 0:
            x2 = torch.cat((x2,x2[:,c-(8-c % 8)-1:c-1,:,:]), dim=1)
            x4 = torch.cat((x4,x4[:,c-(8-c % 8)-1:c-1,:,:]), dim=1)
            x8 = torch.cat((x8,x8[:,c-(8-c % 8)-1:c-1,:,:]), dim=1)
        x2_1 = self.conv2(x2)
        x4_1 = self.conv4(x4)
        x8_1 = self.conv8(x8)

        return x2_1,x4_1,x8_1

class SpaPyBlock(nn.Module):
    def __init__(self, inchannels,outchannels, bias=True):
        super(SpaPyBlock, self).__init__()
        self.scale1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)
        )
        self.scale1_2 = nn.Sequential(
            nn.Upsample(mode='bilinear',scale_factor=1/2),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)
        )
        self.scale1_4 = nn.Sequential(
            nn.Upsample(mode='bilinear',scale_factor=1/4),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)
        )
        self.channel = ChannelAttention(inchannels*3)
        self.out = nn.Sequential(
            nn.Conv2d(inchannels*3, outchannels, kernel_size=3, stride=1, padding=1, groups=1, bias=bias),
            nn.LeakyReLU(0.2)
            )
        

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        y1 = self.scale1(x)
        y2 = self.up2(self.scale1_2(x))
        y3 = self.up4(self.scale1_4(x))
        y = torch.cat((y1,y2,y3),dim=1)
        y = self.channel(y)*y
        y = self.out(y)   
        return y

@register_model('dspnet')
class DSPNet(BaseModel):
    def __init__(self, hschannels, mschannels, bilinear=True):
        super(DSPNet, self).__init__()
        self.bilinear = bilinear

        self.in_conv = nn.Conv2d(150, 32, 1)
        
        self.spe1 = SpePyBlock(32)
        self.spe2 = SpePyBlock(64)
        self.spe3 = SpePyBlock(128)
        self.ds   = nn.Upsample(mode='bilinear',scale_factor=1/2)
        self.mls1  = MLSIF(
                    dim=64, num_blocks=1, dim_head=64, heads=64 // 64)
        self.mls2  = MLSIF(
                    dim=128, num_blocks=1, dim_head=64, heads=128 // 64)
        self.mls3  = MLSIF(
                    dim=256, num_blocks=1, dim_head=64, heads=256 // 64)
        self.inc = DoubleConv(hschannels+mschannels, 64)
        self.down1 = Down(64+32, 128)
        self.down2 = Down(128+64, 256)

        self.up1 = Upm(128+256, 128, bilinear)
        self.up2 = Upm(128+128+64, 64, bilinear)
        self.outc = OutConv(64+64+32, hschannels)

        self.spa1 = SpaPyBlock(mschannels, 32)
        self.spa2 = Downm(32, 64)
        self.spa3 = Downm(64, 128)


    def _forward_implem(self, x,y):
        r = y.size(2) // x.size(2)
        x = F.interpolate(x, scale_factor=r, mode='bicubic', align_corners=False)
        x0 = x
        y1 = self.spa1(y)
        y2 = self.spa2(y1)
        y3 = self.spa3(y2)
        
        x2 = self.in_conv(x)
        z1_2,z1_4,z1_8 = self.spe1(x2,x2,x2)
        z2_2,z2_4,z2_8 = self.spe2(self.ds(z1_2),self.ds(z1_4),self.ds(z1_8))
        z3_2,z3_4,z3_8 = self.spe3(self.ds(z2_2),self.ds(z2_4),self.ds(z2_8))
        
        x1 = self.inc(torch.cat((x,y),dim=1))
        x1 = self.mls1(z1_2,z1_4,z1_8,x1)
        x2 = self.down1(torch.cat((x1, y1), dim=1))
        x2 = self.mls2(z2_2,z2_4,z2_8,x2)
        x3 = self.down2(torch.cat((x2, y2), dim=1))
        x3 = self.mls3(z3_2,z3_4,z3_8,x3)
        x = self.up1(torch.cat((x3, y3),dim=1))
        x = self.up2(torch.cat((x, x2, y2),dim=1))
        logits = self.outc(torch.cat((x, x1, y1),dim=1))

        return logits+x0
    
    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self._forward_implem(ms, pan)
        loss = criterion(sr, gt)
        return sr, loss

    def val_step(self, ms, lms, pan):
        sr = self._forward_implem(ms, pan)
        return sr

    def patch_merge_step(self, ms, lms, pan, hisi=True, split_size=64):
        # all shape is 64
        mms = F.interpolate(ms, size=(split_size // 2, split_size // 2), mode='bilinear', align_corners=True)
        ms = F.interpolate(ms, size=(split_size // 4, split_size // 4), mode='bilinear', align_corners=True)
        if hisi:
            pan = pan[:, :3]
        else:
            pan = pan[:, :1]

        sr = self._forward_implem(ms, pan)

        return sr


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Upsample(mode='bilinear',scale_factor=1/2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Downm(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Upsample(mode='bilinear',scale_factor=1/2),
            SpaPyBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Upm(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)


    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),

        )
    def forward(self, x):
        return self.conv(x)
    
    
if __name__ == '__main__':
    import torch.cuda as tc
    
    C, H, W = 120, 32, 32
    r = 2
    
    net = DSPNet(C, 4).cuda()
    
    lrhsi = torch.randn(1, C, H//r, W//r).cuda()
    rgb = torch.randn(1, 4, H, W).cuda()
    
    net.forward = net._forward_implem
    
    print(net(lrhsi, rgb).shape)
    
    print(tc.memory_summary())
