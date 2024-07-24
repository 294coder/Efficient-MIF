# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 22:26:22 2022
@author: likun
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from model.base_model import BaseModel, register_model

'''
# =================================
# Advanced nn.Sequential
# reform nn.Sequentials and nn.Modules to a single nn.Sequential
# =================================
'''

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)



'''
# =================================
# Downsampler
# =================================
'''

# -------------------------------
# strideconv + relu
# -------------------------------
def downsample_strideconv(
    in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R'
):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride      = int(mode[0])
    mode        = mode.replace(mode[0], 'C')
    down1       = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
    
    return down1


# -------------------------------
# maxpooling + conv + relu
# -------------------------------
def downsample_maxpool(
    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R'
):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool      = int(mode[0])
    mode             = mode.replace(mode[0], 'MC')
    pool             = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail        = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:])
    
    return sequential(pool, pool_tail)


# -------------------------------
# averagepooling + conv + relu
# -------------------------------
def downsample_avgpool(
    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R'
):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool      = int(mode[0])
    mode             = mode.replace(mode[0], 'AC')
    pool             = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail        = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:])
    
    return sequential(pool, pool_tail)

 


'''
# =================================
# Upsampler
# =================================
'''

# -------------------------------
# convTranspose + relu
# -------------------------------
def upsample_convtranspose(
    in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R'
):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride      = int(mode[0])
    
    mode        = mode.replace(mode[0], 'T')
    up1         = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
    return up1

# -------------------------------
# conv + subp + relu
# -------------------------------
def upsample_pixelshuffle(
    in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R'
):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode)
    return up1

# -------------------------------
# nearest_upsample + conv + relu
# -------------------------------
def upsample_upconv(
    in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R'
):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
        
    mode   = mode.replace(mode[0], uc)
    up1    = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode)
    return up1


'''
# ===================================
# Useful blocks
# --------------------------------
# conv (+ normaliation + relu)
# resblock (ResBlock)

# ===================================
'''

def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR'):
    L = []
    for t in mode:
        if   t == 'C':
            L.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                )
            )
        elif t == 'T':
            L.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                )
            )
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))   
            # https://www.zhihu.com/question/68730628/answer/607608890
            
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=False))          
            
        elif t == '1':
            L.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias
                )
            )
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
            
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
            
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
              
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)
 
# -------------------------------
# Res Block: x + conv(relu(conv(x)))
# -------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC'):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        
        if mode[0] in ['R','L']:  mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)

    def forward(self, x):
        
        res = self.res(x)
        return x + res

# -------------------------------
# Res Block: x + conv(relu(conv(x)))
# -------------------------------
class ResBlock_ablation1(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC'):
        super(ResBlock_ablation1, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        
        if mode[0] in ['R','L']:  mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)

    def forward(self, x):
        
        res = self.res(x)
        return res

        
""" ------------ main fusionnet ------------ """
@register_model('hpmnet')
class fusionnet(BaseModel):
    
    def __init__(
        self, n_iter=6, h_nc=64, in_c=32, out_c=31, m_c=3, nc=[80, 160, 320], nb=1,
        act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose"):
 
        super(fusionnet, self).__init__()
        self.n = n_iter
        self.z = Z_SubNet  (out_c =out_c, m_c=m_c)
        self.h = H_hyperNet(in_c  =3 , out_c=n_iter*3, channel=h_nc)  
        self.x = X_PriorNet(
            in_c=in_c, out_c=out_c, nc=nc, nb=nb,
            act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        
        lamda_m = 0.01  
        lamda_p = 0.01  
        lamda_m = torch.tensor(lamda_m).float().view([1, 1, 1, 1])
        lamda_p = torch.tensor(lamda_p).float().view([1, 1, 1, 1])  
        [lamda_m, lamda_p] = [el for el in [lamda_m, lamda_p]] 
        self.register_buffer('lamda_m', lamda_m)
        self.register_buffer('lamda_p', lamda_p)
      
    def _forward_implem(self, Y_h, Y_m, Y_p, lamda_m=None, lamda_p=None):
        if lamda_m is None:
            lamda_m = self.lamda_m
        if lamda_p is None:
            lamda_p = self.lamda_p
        
        # Initialization    
        mu     = 0.01
        mu     = torch.tensor(mu).float().view([1, 1, 1, 1]).to(lamda_p)
        beta   = torch.sqrt(mu*lamda_p) # ρ = α.mu.lamda_p
        # Y_p = Y_h.mean(0, keepdim=True)
        rate_h = Y_p.shape[2]//Y_h.shape[2]
        x      = F.interpolate(Y_h, scale_factor=float(rate_h), mode ='bilinear')
        # hyper-parameter
        hypers = self.h(torch.cat((mu, lamda_m, beta), dim=1))
        # unfolding    
        for i in range(self.n): 
            z          = self.z(x, Y_h, Y_m, hypers[:, i, ...], hypers[:, i+self.n, ...] )     
            x          = self.x(z, Y_p,      hypers[:, i+self.n*2, ...])
        return x 
    
    def train_step(self, ms, lms, pan, gt, criterion):
        pan_downsample = F.interpolate(pan, size=(pan.shape[2]//2, pan.shape[3]//2), mode='bilinear', align_corners=True)
        sr = self._forward_implem(ms, pan_downsample, pan)
        loss = criterion(sr, gt)
        
        return sr, loss
    
    def val_step(self, ms, lms, pan):
        pan_downsample = F.interpolate(pan, size=(pan.shape[2]//2, pan.shape[3]//2), mode='bilinear', align_corners=True)
        sr = self._forward_implem(ms, pan_downsample, pan)
        return sr
    
    def patch_merge_step(self, ms, lms, pan, hisi=True, split_size=64):
        # all shape is 64
        pan_downsample = F.interpolate(pan, size=(pan.shape[2]//2, pan.shape[3]//2), mode='bilinear', align_corners=True)
        mms = F.interpolate(ms, size=(split_size // 2, split_size // 2), mode='bilinear', align_corners=True)
        ms = F.interpolate(ms, size=(split_size // 4, split_size // 4), mode='bilinear', align_corners=True)
        if hisi:
            pan = pan[:, :3]
        else:
            pan = pan[:, :1]

        sr = self._forward_implem(ms, pan_downsample, pan)

        return sr
        
        


""" -------------- -------------- --------------
# (1) Intermediate estimate Z; Gradient-based opt 
# z_k = x_{k-1} - mu{grad(f(.))}
-------------- -------------- -------------- """

class Z_SubNet(nn.Module):
    def __init__(self, out_c =31, m_c=3):
        super(Z_SubNet, self).__init__()

        self.Rm_conv  = conv(out_c, m_c, bias=False, mode='1')
        self.RmT_conv = conv(m_c, out_c, bias=False, mode='1')

    def forward(self, x, Y_h, Y_m, mu, lamda_m):
        rate_h, rate_m  = x.shape[2]//Y_h.shape[2], x.shape[2]//Y_m.shape[2]

        XS1         = F.interpolate(x        , scale_factor = float(1.0/rate_h) , mode ='bilinear')  
        Diff_S1T    = F.interpolate((XS1-Y_h), scale_factor = float(rate_h)     , mode ='bilinear')  
        RXS2        = self.Rm_conv( F.interpolate(x         , scale_factor = float(1.0/rate_m) , mode ='bilinear'))  
        RTDiff_S2T  = self.RmT_conv(F.interpolate((RXS2-Y_m), scale_factor = float(rate_m)     , mode ='bilinear'))
        Zest = x - mu*Diff_S1T - mu*lamda_m*RTDiff_S2T
        return Zest 


""" -------------- -------------- --------------
# (2) Observation Variable X  && Prior module
#     X  --> Prior pf W   --> obtained X
#     Transfer2 Prior(W)  --> X = W-Rp^Y_p
-------------- -------------- -------------- """ 
class X_PriorNet(nn.Module):
    def __init__(
        self, in_c=32, out_c=31, nc=[80, 160, 320], nb=1, 
        act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        
        super(X_PriorNet, self).__init__() 
        # downsample
        if downsample_mode   == 'avgpool':    downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':    downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv': downsample_block = downsample_strideconv
        else: raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_head  = conv(in_c, nc[0], bias=False, mode='C')
        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], 
            downsample_block(nc[0], nc[1], bias=False, mode='2')
        )
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], 
            downsample_block(nc[1], nc[2], bias=False, mode='2')
        )
        self.m_body  = sequential(
            *[ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )
  
        # upsample
        if upsample_mode   == 'upconv':         upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':   upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':  upsample_block = upsample_convtranspose
        else: raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up2  = sequential(
            upsample_block(nc[2], nc[1], bias=False, mode='2'), 
            *[ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )
        self.m_up1  = sequential(
            upsample_block(nc[1], nc[0], bias=False, mode='2'), 
            *[ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )
        self.m_tail = conv(nc[0], out_c, bias=False, mode='C')


        self.Rp_conv     = nn.Sequential(
                            nn.Conv2d(out_c,  out_c,  3, stride=1, padding=1),
                            nn.ReLU(inplace = True),
                            nn.Conv2d(out_c,  out_c,  3, stride=1, padding=1),)
        self.Rp_hat_conv = nn.Sequential(
                            nn.Conv2d(out_c,  out_c,  3, stride=1, padding=1),
                            nn.ReLU(inplace = True),
                            nn.Conv2d(out_c,  out_c,  3, stride=1, padding=1),)

    def forward(self, Z, Y_p, beta):
        Y_p_copy = Y_p.repeat(1, Z.shape[1], 1, 1) 
        Denoi    = self.Rp_conv(Z) - Y_p_copy  
        Betas    = beta.repeat(Z.size(0), 1, Z.size(2), Z.size(3))         
        W = torch.cat((Denoi ,Betas),  dim = 1)

        ht, wt = W.size()[-2:]
        paddingBottom = int(np.ceil(ht/8)*8-ht)
        paddingRight  = int(np.ceil(wt/8)*8-wt)
        W = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(W)
        W1 = self.m_head(W)
        W2 = self.m_down1(W1)
        W3 = self.m_down2(W2) 
        W  = self.m_body(W3) 
        W  = self.m_up2(W+W3)
        W  = self.m_up1(W+W2)
        W  = self.m_tail(W+W1)
        W = W[..., :ht, :wt]

        Xest = self.Rp_hat_conv(Y_p_copy + W) 
        return Xest
    
        
    
""" -------------- -------------- --------------
# (3) Hyper-parameter module
-------------- -------------- -------------- """ 
class H_hyperNet(nn.Module):
    def __init__(self, in_c=3, out_c=6*3, channel=64):
        super(H_hyperNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_c, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_c, 1, padding=0, bias=True),
                nn.Softplus())
    
    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x
    
if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    
    device = 'cuda:0'
    
    net = fusionnet(
        n_iter=5,
        h_nc=64,
        in_c=8+1,
        out_c=8,
        m_c=3,
        nc=[80, 160, 320],
        nb=1,
        act_mode="R",
        downsample_mode='strideconv',
        upsample_mode='convtranspose'
    ).to(device)
    
    lr_hsi = torch.randn(1, 8, 16, 16).to(device)
    hr_msi = torch.randn(1, 1, 32, 32).to(device)
    pan = torch.randn(1, 1, 64, 64).to(device)
    
    lamda_m = 0.01  
    lamda_p = 0.01  
    lamda_m = torch.tensor(lamda_m).float().view([1, 1, 1, 1])
    lamda_p = torch.tensor(lamda_p).float().view([1, 1, 1, 1])
    [lamda_m, lamda_p] = [el.to(device) for el in [lamda_m, lamda_p]]
    
    
    # num_p = 0
    # for p in net.parameters():
    #     num_p += p.numel()
        
    # print(num_p / 1000_000)
    
    
    net.forward = net._forward_implem
    
    print(net(lr_hsi, hr_msi, pan, lamda_m, lamda_p).shape)
    
    print(flop_count_table(
        FlopCountAnalysis(
            net,
            (lr_hsi, hr_msi, pan, lamda_m, lamda_p)
    ))
    )

    