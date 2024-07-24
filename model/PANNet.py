import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_model import BaseModel, register_model
from model.module.resblock import Resblock
from model.module.stem import GaussianHighPassModule
from utils import variance_scaling_initializer


@register_model("pannet")
class VanillaPANNet(BaseModel):
    def __init__(self, in_c, pan_c=1, hidden_c=32, factor=2):
        super(VanillaPANNet, self).__init__()
        # self.ghpm = GaussianHighPassModule(in_c + 1, 64, 64)
        self.conv1 = nn.Conv2d(
            in_channels=in_c + pan_c,
            out_channels=hidden_c,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # _res_block_params = dict(in_c=hidden_c, out_c=hidden_c, ksize=(3, 3), stride=1, pad=1)
        self.res1 = Resblock(hidden_c)
        self.res2 = Resblock(hidden_c)
        self.res3 = Resblock(hidden_c)
        self.res4 = Resblock(hidden_c)

        self.conv2 = nn.Conv2d(
            in_channels=hidden_c,
            out_channels=in_c,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.relu = nn.ReLU()

        if factor == 4:
            ## for factor of 4
            self.deconv = nn.ConvTranspose2d(
                in_channels=in_c,
                out_channels=in_c,
                kernel_size=8,
                stride=4,
                padding=2,
                bias=True,
            )
        elif factor == 2:
            ## for factor of 2
            self.deconv = nn.ConvTranspose2d(
                in_channels=in_c,
                out_channels=in_c,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            )
        else:
            raise ValueError("@factor must be 2 or 4")

        self.backbone = nn.Sequential(self.res1, self.res2, self.res3, self.res4)

        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):  # initialization for Conv2d
            # print("nn.Conv2D is initialized by variance_scaling_initializer")
            variance_scaling_initializer(m.weight)  # method 1: initialization
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):  # initialization for BN
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):  # initialization for nn.Linear
            # variance_scaling_initializer(m.weight)
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _forward_implem(self, ms, pan):
        output_deconv = self.deconv(ms)  # Bsx8x64x64

        input = torch.cat([output_deconv, pan], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone! # Bsx32x64x64

        output = self.conv2(rs)  # Bsx8x64x64
        return output

    def train_step(self, ms, lms, pan, gt, criterion):
        hr = self._forward_implem(ms, pan)
        hr += lms
        loss = criterion(hr, gt)
        return hr, loss

    @torch.no_grad()
    def val_step(self, ms, lms, pan):
        hr = self._forward_implem(ms, pan)
        hr += lms
        return hr


if __name__ == "__main__":
    pannet = VanillaPANNet(150, 4, 64)
    x = torch.randn(1, 150, 40, 40)
    y = torch.randn(1, 4, 80, 80)
    print(pannet._forward_implem(x, y).shape)
