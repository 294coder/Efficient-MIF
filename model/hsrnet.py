# GPL License
# Copyright (C) UESTC
# All Rights Reserved
#
# @Time    : 2023/5/21 22:51
# @Author  : Xiao Wu
# @reference:
#
import torch
from torch import nn

from model.base_model import BaseModel, register_model


def _phase_shift(I, r):
    bsize, c, h, w = I.shape
    bsize = I.shape[0]
    X = torch.reshape(I, (bsize, r, r, h, w))
    X = torch.chunk(X, chunks=w, dim=3)  # 在w通道上分成了w份， 将每一维分成了1
    # tf.squeeze删除axis上的1，然后在第三通道 即r通道上 将w个小x重新级联变成r * w
    X = torch.concat([torch.squeeze(x, dim=3) for x in X], 1)  # 最终变成 bsize, h, r * w, r
    X = torch.chunk(X, chunks=h, dim=3)
    X = torch.cat([torch.squeeze(x, dim=3) for x in X], 2)

    return torch.reshape(X, (bsize, 1, h * r, w * r))  # 最后变成这个shape


@register_model("hsrnet")
class HSRNet(BaseModel):
    def __init__(self, spectral_num, rgb_num, num_res=6, num_feature=64, scale=4):
        super().__init__()

        self.scale = scale
        
        self.ms_ch_attn = nn.Sequential(
            nn.Conv2d(spectral_num, 1, kernel_size=1, stride=1),
            nn.Conv2d(1, spectral_num, kernel_size=1, stride=1),
        )

        self.rgb_spa_attn = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        # (64 - k + 2*p)/4 + 1 = 32
        # 58/4 + p/2 = 32
        # ((64//scale - 1) * scale + k - 64) // 2
        self.rgb_conv = nn.Conv2d(
            rgb_num, rgb_num, kernel_size=6, stride=scale, padding=2 #((64//scale-1)*scale+6-64)//scale
        )  # kernel_size=6, stride=4

        self.rs_conv1 = nn.Conv2d(
            spectral_num + rgb_num,
            spectral_num * scale * scale,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.rs_conv2 = nn.Conv2d(
            spectral_num + rgb_num, num_feature, kernel_size=3, stride=1, padding=1
        )
        self.rs_blocks = nn.ModuleList()
        for i in range(num_res):
            self.rs_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        num_feature, num_feature, kernel_size=3, stride=1, padding=1
                    ),
                    nn.Conv2d(
                        num_feature, num_feature, kernel_size=3, stride=1, padding=1
                    ),
                )
            )
        self.rs_conv3 = nn.Conv2d(
            num_feature, spectral_num, kernel_size=3, stride=1, padding=1
        )

    def _forward_implem(self, ms, RGB):
        gap_ms_c = torch.mean(ms, dim=[2, 3], keepdim=True)
        CA = self.ms_ch_attn(gap_ms_c)
        gap_RGB_s = torch.mean(RGB, dim=1, keepdim=True)
        SA = self.rgb_spa_attn(gap_RGB_s)
        rgb = self.rgb_conv(RGB)
        # rslice, gslice, bslice = torch.split(rgb, [1, 1, 1], dim=1)
        c_slices = rgb.chunk(rgb.size(1), dim=1)
        ms_c = ms.size(1)
        msp1, msp2 = torch.split(ms, [ms_c//2, ms_c-ms_c//2], dim=1)
        if rgb.size(1) == 3:
            ms = torch.cat([c_slices[0], msp1, c_slices[1], msp2, c_slices[2]], dim=1)
        elif rgb.size(1) == 4:
            ms = torch.cat([c_slices[0], msp1, c_slices[1], msp2, c_slices[2], c_slices[3]], dim=1)
        rs = self.rs_conv1(ms)
        Xc = torch.chunk(rs, chunks=ms_c, dim=1)
        rs = torch.cat(
            [_phase_shift(x, self.scale) for x in Xc], 1
        )  # each of x in Xc is r * r channel 分别每一个通道变为r*r
        # Rslice, Gslice, Bslice = torch.split(RGB, [1, 1, 1], dim=1)
        C_slices = RGB.chunk(RGB.size(1), dim=1)
        Msp1, Msp2 = torch.split(rs, [ms_c//2, ms_c-ms_c//2], dim=1)
        if rgb.size(1) == 3:
            rs = torch.cat([C_slices[0], Msp1, C_slices[1], Msp2, C_slices[2]], dim=1)
        elif rgb.size(1) == 4:
            rs = torch.cat([C_slices[0], Msp1, C_slices[1], Msp2, C_slices[2], C_slices[3]], dim=1)
        # rs = torch.concat([Rslice, Msp1, Gslice, Msp2, Bslice], dim=1)
        rs = self.rs_conv2(rs)
        for rs_block in self.rs_blocks:
            rs = rs + rs_block(rs)
        rs = SA * rs
        rs = self.rs_conv3(rs)
        rs = CA * rs
        return rs, CA, SA

    def train_step(self, lrhsi, up_hs, rgb, gt, criterion):
        rs, *_ = self._forward_implem(lrhsi, rgb)
        loss = criterion(rs, gt)

        return rs, loss

    @torch.no_grad()
    def val_step(self, lrhsi, up_hs, rgb):
        rs, *_ = self._forward_implem(lrhsi, rgb)

        return rs


if __name__ == "__main__":
    ms = torch.randn([1, 31, 8, 8])
    RGB = torch.randn([1, 3, 64, 64])
    model = HSRNet(31, 3, scale=8)
    # print(model._forward_implem(ms, RGB)[0].shape)
    
    from fvcore.nn import flop_count_table, FlopCountAnalysis, parameter_count_table

    model.forward = model._forward_implem
    flops = FlopCountAnalysis(model, (ms, RGB))
    print(flop_count_table(flops, max_depth=3))
    print(flops.total() / 1e9, 'G')
