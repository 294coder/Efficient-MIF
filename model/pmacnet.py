import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

from model.base_model import BaseModel, register_model


class MRFAB(BaseModel):
    def __init__(self, ch_in=64, ch_out=256):
        super(MRFAB, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv1_1 = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=2, padding_mode='reflect', dilation=2)
        self.conv1_2 = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=3, padding_mode='reflect', dilation=3)
        self.conv1_3 = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=4, padding_mode='reflect', dilation=4)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(ch_out, ch_in, kernel_size=1, stride=1)
        self.ln = nn.LayerNorm([ch_out], elementwise_affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.conv1_1(x)
        out2 = self.conv1_2(x)
        out3 = self.conv1_3(x)
        out = torch.cat((out, out1, out2, out3), dim=1)
        out = self.ln(out.permute(0, 2, 3, 1))
        out = out.permute(0, 3, 1, 2)
        out = F.gelu(self.conv2(out))
        out = self.conv3(out)
        out = out + x
        return out


class Det_Reg(nn.Module):
    def __init__(self, ms_inp_ch=4, num_layers=2, latent_dim=64):
        super(Det_Reg, self).__init__()
        blks = []
        for i in range(num_layers):
            blks.append(MRFAB(latent_dim, 4 * latent_dim))
        blks1 = []
        for i in range(num_layers):
            blks1.append(MRFAB(latent_dim, 4 * latent_dim))
        self.detail_reg = nn.Sequential(
            nn.Conv2d(ms_inp_ch, latent_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            *blks,
            nn.Conv2d(latent_dim, ms_inp_ch, kernel_size=3, stride=1, padding=1, padding_mode='reflect'))
        self.roi_reg = nn.Sequential(
            nn.Conv2d(ms_inp_ch, latent_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            *blks1,
            nn.Conv2d(latent_dim, ms_inp_ch, kernel_size=3, stride=1, padding=1, padding_mode='reflect'))
        self.sm = nn.Softmax(dim=1)

    def forward(self, ms, pan):
        ms_df = self.detail_reg(ms)
        pan_df = self.roi_reg(pan)
        q_out = torch.mul((self.sm(torch.mul(ms_df, pan_df))), pan_df)
        return q_out


@register_model('pmacnet')
class PMACNet(BaseModel):
    def __init__(self, ms_inp_ch=8, num_layers=2, latent_dim=64):
        super(PMACNet, self).__init__()
        self.ms_inp_ch = ms_inp_ch
        self.Det_Reg = Det_Reg(ms_inp_ch, num_layers, latent_dim)

    def _forward_implem(self, ms, pan):
        pan = pan.repeat(1, self.ms_inp_ch, 1, 1)
        pan_d = pan - ms
        out = self.Det_Reg(ms, pan_d)
        return out

    def train_step(self, ms, lms, pan, gt, criterion):
        pred = self._forward_implem(lms, pan)
        loss = criterion(pred, gt)
        return pred, loss

    @torch.no_grad()
    def val_step(self, ms, lms, pan):
        pred = self._forward_implem(lms, pan)
        return pred

def summaries(model, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 64, 64), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)


# model = PMACNet().cuda()
# summaries(model, grad=True)

if __name__ == '__main__':
    lms = torch.randn(1, 4, 64, 64).cuda()
    pan = torch.randn(1, 1, 64, 64).cuda()
    model = PMACNet().cuda()
    out = model._forward_implem(lms, pan)

    print(out.shape)
    
    # 2peiwuta qb on 2
    
    # y2k8paq1 gf2 on 6
