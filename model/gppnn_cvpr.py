# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_model import BaseModel, register_model



def upsample(x, h, w):
    return F.interpolate(x, size=[h,w], mode='bicubic', align_corners=True)

class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels,out_channels, 3, padding=1, bias=False)
        self.relu  = nn.ReLU(True)
        
    def forward(self, x):
        x = x+self.conv2(self.relu(self.conv1(x)))
        return x
    
class BasicUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3):
        super(BasicUnit, self).__init__()
        p = kernel_size//2
        self.basic_unit = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=p, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=p, bias=False)
            )

    def forward(self, input):
        return self.basic_unit(input)

# LRBlock is called MSBlock in our paper
class LRBlock(nn.Module):
    def __init__(self,
                 ms_channels,
                 n_feat):
        super(LRBlock, self).__init__()
        self.get_LR = BasicUnit(ms_channels, n_feat, ms_channels)
        self.get_HR_residual = BasicUnit(ms_channels, n_feat, ms_channels)
        self.prox = BasicUnit(ms_channels, n_feat, ms_channels)
        
    def forward(self, HR, LR):
        _,_,M,N = HR.shape
        _,_,m,n = LR.shape
        
        LR_hat = upsample(self.get_LR(HR), m, n)
        LR_Residual = LR - LR_hat
        HR_Residual = upsample(self.get_HR_residual(LR_Residual), M, N)
        HR = self.prox(HR + HR_Residual)
        return HR
        
class PANBlock(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat, 
                 kernel_size):
        super(PANBlock, self).__init__()
        self.get_PAN = BasicUnit(ms_channels, n_feat, pan_channels, kernel_size)
        self.get_HR_residual = BasicUnit(pan_channels, n_feat, ms_channels, kernel_size)
        self.prox = BasicUnit(ms_channels, n_feat, ms_channels, kernel_size)
        
    def forward(self, HR, PAN):
        PAN_hat = self.get_PAN(HR)
        PAN_Residual = PAN - PAN_hat
        HR_Residual = self.get_HR_residual(PAN_Residual)
        HR = self.prox(HR + HR_Residual)
        return HR

@register_model('gppnn_cvpr')
class GPPNN(BaseModel):
    def __init__(self, 
                 ms_channels,
                 pan_channels,
                 n_feat,
                 n_layer):
        super(GPPNN, self).__init__()
        self.lr_blocks = nn.ModuleList([LRBlock(ms_channels, n_feat) for i in range(n_layer)])
        self.pan_blocks = nn.ModuleList([PANBlock(ms_channels, pan_channels, n_feat, 1) for i in range(n_layer)])

        
    def _forward_implem(self, ms, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w] 
        # pan - high-resolution panchromatic image [N,1,H,W] 
        if type(pan)==torch.Tensor:
            pass
        elif pan==None:
            raise Exception('User does not provide pan image!')
        _,_,m,n = ms.shape
        _,_,M,N = pan.shape
        HR = upsample(ms, M, N)
        
        for i in range(len(self.lr_blocks)):
            HR = self.lr_blocks[i](HR, ms)
            HR = self.pan_blocks[i](HR, pan)
            
        return HR
    
    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self._forward_implem(ms, pan)
        loss = criterion(sr, gt)

        return sr, loss
    
    @torch.no_grad()
    def val_step(self, ms, lms, pan):
        sr = self._forward_implem(ms, pan)

        return sr
    
    
if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis
    
    net = GPPNN(102, 3, 64, 8)
    ms = torch.randn(1,102,16,16)
    pan = torch.randn(1,1,64,64)
    
    net.forward = net._forward_implem
    
    print(flop_count_table(FlopCountAnalysis(net, (ms, pan))))
    
    # print(net.val_step(ms, ms, pan).shape)