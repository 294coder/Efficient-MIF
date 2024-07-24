import torch
import torch.nn as nn
import numpy as np
import math

from model.base_model import BaseModel, register_model
from model.module.resblock import Resblock
from utils import variance_scaling_initializer, loss_with_l2_regularization


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                # print("initial nn.Conv2d with var_scale_new: ", m)
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)  # method 1: initialization
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


@register_model('fusionnet')
class FusionNet(BaseModel):
    def __init__(self, spectral_num, channel=32):
        super(FusionNet, self).__init__()
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.spectral_num = spectral_num

        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

        init_weights(self.backbone, self.conv1, self.conv3)  # state initialization, important!
        self.apply(init_weights)

    def _forward_implem(self, x, y):  # x= lms; y = pan

        pan_concat = y.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_concat, x)  # Bsx8x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx8x64x64

        return output  # lms + outs

    def train_step(self, ms, lms, pan, gt, criterion):
        res = self._forward_implem(lms, pan)
        sr = lms + res  # output:= lms + hp_sr
        # reg = loss_with_l2_regularization()
        loss = criterion(sr, gt)
        # loss = reg(loss, self)
        return sr, loss

    def val_step(self, ms, lms, pan):
        res = self._forward_implem(lms, pan)
        sr = lms + res  # output:= lms + hp_sr

        return sr


if __name__ == '__main__':
    fusionnet = FusionNet(8, 32)
    x = torch.randn(1, 8, 64, 64)
    y = torch.randn(1, 1, 64, 64)
    print(fusionnet._forward_implem(x, y).shape)
