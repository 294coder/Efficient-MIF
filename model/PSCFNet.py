from collections import OrderedDict
import sys
from typing import Tuple, Union
from torch import Tensor
import torch.nn as nn
from torch.nn import Sequential, MaxPool2d, Conv2d, ReLU, PReLU
import torch
import math
import torch.nn.functional as F

from model.base_model import BaseModel, register_model


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def activation(act_type='prelu', slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!' % act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] Normalization layer [%s] is not implemented!' % norm_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict' % sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# ------build blocks------ #
def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,
              act_type='prelu', norm_type='bn', pad_type='zero'):
    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias)

    act = activation(act_type) if act_type else None
    n = norm(out_channels, norm_type) if norm_type else None
    return sequential(p, conv, n, act)


def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, padding=0, act_type='relu',
                norm_type='bn', pad_type='zero'):
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
    act = activation(act_type) if act_type else None
    n = norm(out_channels, norm_type) if norm_type else None
    return sequential(p, deconv, n, act)


class CFB(nn.Module):
    def __init__(self, norm_type):
        super(CFB, self).__init__()
        upscale_factor = 2
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8

        self.num_groups = 3
        num_features = 64
        act_type = 'prelu'

        self.compress_in1 = ConvBlock(3 * num_features, num_features, kernel_size=1, act_type=act_type,
                                      norm_type=norm_type)

        self.compress_in2 = ConvBlock(4 * num_features, num_features, kernel_size=1, act_type=act_type,
                                      norm_type=norm_type)

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        self.re_guide = ConvBlock(2 * num_features, num_features, kernel_size=1, act_type=act_type,
                                  norm_type=norm_type)
        self.re_guide1 = ConvBlock(3 * num_features, num_features, kernel_size=1, act_type=act_type,
                                   norm_type=norm_type)
        # self.re_guide = ConvBlock(3 * num_features, num_features, kernel_size=1, act_type=act_type,
        #                           norm_type=norm_type)

        for idx in range(self.num_groups):
            self.upBlocks.append(
                DeconvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                            act_type=act_type, norm_type=norm_type))
            self.downBlocks.append(
                ConvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                          act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(
                    ConvBlock(num_features * (idx + 1), num_features, kernel_size=1, stride=1, act_type=act_type,
                              norm_type=norm_type))
                self.downtranBlocks.append(
                    ConvBlock(num_features * (idx + 1), num_features, kernel_size=1, stride=1, act_type=act_type,
                              norm_type=norm_type))

        self.compress_out = ConvBlock(self.num_groups * num_features, num_features, kernel_size=1, act_type=act_type,
                                      norm_type=norm_type)

    def forward(self, g1, g2, a):
        # x = torch.cat((f_in, g1), dim=1)
        # x = torch.cat((x, g2), dim=1)
        x = torch.cat((g1, g2), dim=1)
        if a < 1:
            x = self.compress_in1(x)
        else:
            x = self.compress_in2(x)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)
            if idx > 0:
                LD_L = self.uptranBlocks[idx - 1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx - 1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            if a < 1:
                if idx == 2:
                    x_mid = torch.cat((LD_L, g2), dim=1)
                    LD_L = self.re_guide(x_mid)
            else:
                if idx == 2:
                    x_mid = torch.cat((LD_L, g2), dim=1)
                    LD_L = self.re_guide1(x_mid)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)
        output = self.compress_out(output)

        return output


@register_model('pscfnet')
class PSCF_Net(BaseModel):
    def __init__(self, input_channel=4, norm_type=None):
        super(PSCF_Net, self).__init__()

        self.FEB_MS = nn.Sequential(
            ConvBlock(in_channels=4, out_channels=128, kernel_size=3, act_type=None, norm_type=norm_type),
            ConvBlock(in_channels=128, out_channels=64, kernel_size=1, norm_type=norm_type)
        )

        self.FEB_L = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=128, kernel_size=7, act_type=None, norm_type=norm_type),
            ConvBlock(in_channels=128, out_channels=64, kernel_size=1, norm_type=norm_type)
        )

        self.FEB_H = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=128, kernel_size=3, act_type=None, norm_type=norm_type),
            ConvBlock(in_channels=128, out_channels=64, kernel_size=1, norm_type=norm_type)
        )

        self.MS1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,
                      out_channels=64,
                      kernel_size=1,
                      stride=1),
            nn.PReLU(),
        )

        self.fusion1 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=input_channel,
                      kernel_size=1,
                      stride=1),
            nn.Tanh(),

        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

            self.num_cfbs = input_channel

            self.CFBs_1 = []
            self.CFBs_2 = []

            for i in range(self.num_cfbs):
                cfb_pan = 'cfb_pan{}'.format(i)
                cfb_ms = 'cfb_ms{}'.format(i)
                cfb_1 = CFB(norm_type).cuda()
                cfb_2 = CFB(norm_type).cuda()
                setattr(self, cfb_pan, cfb_1)
                self.CFBs_1.append(getattr(self, cfb_pan))
                setattr(self, cfb_ms, cfb_2)
                self.CFBs_2.append(getattr(self, cfb_ms))

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def _forward_implem(self, x_pan, x_lr):

        pan_H = self.FEB_H(x_pan)
        pan_L = self.FEB_L(x_pan)

        ms = self.MS1(x_lr)
        ms_L = self.FEB_MS(x_lr)

        out_DL1 = torch.cat((pan_H, ms), dim=1)

        # Coupled feedback block
        CF_DL = [out_DL1]
        CF_MS = [ms_L]
        CF_pan = [pan_L]

        for i in range(self.num_cfbs):
            if i == 0:
                CF_pan.append(self.CFBs_1[i](CF_DL[i], CF_pan[i], i))
                CF_MS.append(self.CFBs_2[i](CF_DL[i], CF_MS[i], i))
            else:
                a = torch.cat((CF_MS[i], CF_MS[0]), dim=1)
                b = torch.cat((CF_pan[i], CF_pan[0]), dim=1)
                CF_pan.append(self.CFBs_1[i](a, b, i))
                CF_MS.append(self.CFBs_2[i](a, b, i))

        out = torch.cat((CF_pan[self.num_cfbs], CF_MS[self.num_cfbs]), dim=1)

        restore = self.fusion1(out)

        img = restore * (2 ** 11)

        return img

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self._forward_implem(pan, lms)

        loss = criterion(sr, gt)

        return sr, loss

    def val_step(self, ms, lms, pan):
        sr = self._forward_implem(pan, lms)

        return sr


if __name__ == '__main__':
    from torch.cuda import memory_summary

    device = 'cuda:1'

    in_c = 8
    net = PSCF_Net(in_c).to(device)

    ms = torch.randn(1, in_c, 16, 16).to(device)
    lms = torch.randn(1, in_c, 64, 64).to(device)
    pan = torch.randn(1, 1, 64, 64).to(device)
    sr = torch.randn(1, in_c, 64, 64).to(device)

    out = net._forward_implem(pan, lms)

    print(out.shape)

    loss = nn.MSELoss()(out, sr)
    loss.backward()

    print(memory_summary())
