import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from model.base_model import register_model, BaseModel

def spatial_edge(x):
    edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]
    edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]

    return edge1, edge2

def spectral_edge(x):
    edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]

    return edge

@register_model('ssrnet')
class SSRNET(BaseModel):
    def __init__(self, 
                 arch,
                 n_bands,
                 rgb_channel,
                 scale_ratio=None,
                 n_select_bands=None):
        """Load the pretrained ResNet and replace top fc layer."""
        super(SSRNET, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.arch = arch
        self.n_select_bands = n_select_bands
        self.weight = nn.Parameter(torch.tensor([0.5]))

        self.conv_fus = nn.Sequential(
                  nn.Conv2d(n_bands+rgb_channel, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.conv_spat = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.conv_spec = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )

    def lrhr_interpolate(self, x_lr, x_hr):
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        # gap_bands = self.n_bands / (self.n_select_bands-1.0)
        # for i in range(0, self.n_select_bands-1):
        #     x_lr[:, int(gap_bands*i), ::] = x_hr[:, i, ::]
        # x_lr[:, int(self.n_bands-1), ::] = x_hr[:, self.n_select_bands-1, ::]


        return x_lr

    def spatial_edge(self, x):
        edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]
        edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]

        return edge1, edge2

    def spectral_edge(self, x):
        edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]

        return edge

    def _forward_implem(self, x, x_hr):
        # x = self.lrhr_interpolate(x_lr, x_hr)
        # x = x.cuda()
        x = torch.cat((x, x_hr), 1)
        x = self.conv_fus(x)

        if self.arch == 'SSRNET':
            x_spat = x + self.conv_spat(x)
            spat_edge1, spat_edge2 = self.spatial_edge(x_spat) 

            x_spec = x_spat + self.conv_spec(x_spat)
            spec_edge = self.spectral_edge(x_spec)

            x = x_spec

        elif self.arch == 'SpatRNET':
            x_spat = x + self.conv_spat(x)

            spat_edge1, spat_edge2 = self.spatial_edge(x_spat) 
            x_spec = x
            spec_edge = self.spectral_edge(x_spec)

        elif self.arch == 'SpecRNET':
            x_spat = x
            spat_edge1, spat_edge2 = self.spatial_edge(x_spat) 

            x_spec = x + self.conv_spec(x)
            spec_edge = self.spectral_edge(x_spec) 

            x = x_spec

        return x, x_spat, x_spec, spat_edge1, spat_edge2, spec_edge

    def train_step(self, lrhsi, up_hs, rgb, gt, criterion):

        # gt, up_hs, hs, rgb = data['gt'].cuda(), data['up'].cuda(), \
        #                    data['lrhsi'].cuda(), data['rgb'].cuda()
        out, out_spat, out_spec, edge_spat1, edge_spat2, edge_spec = self._forward_implem(up_hs, rgb)
        ref_edge_spat1, ref_edge_spat2 = spatial_edge(gt)
        ref_edge_spec = spectral_edge(gt)
        
        # reorganize the loss function
        loss_fus, _ = criterion(out, gt)
        loss_spat, _ = criterion(out_spat, gt)
        loss_spec, _ = criterion(out_spec, gt)
        loss_spec_edge, _ = criterion(edge_spec, ref_edge_spec)
        loss_spat_edge1, _ = criterion(edge_spat1, ref_edge_spat1)
        loss_spat_edge2, _ = criterion(edge_spat2, ref_edge_spat2)
        loss_spat_edge = 0.5 * (loss_spat_edge1 + loss_spat_edge2)
        
        # _loss_ret = {
        #     'loss_fus': loss_fus,
        #     'loss_spat': loss_spat,
        #     'loss_spec': loss_spec,
        #     'loss_spec_edge': loss_spec_edge,
        # }
        
        if self.arch == 'SpatRNET':
            loss = loss_spat + loss_spat_edge
            loss_ret = {'loss_spat': loss_spat, 'loss_spat_edge': loss_spat_edge}
        elif self.arch == 'SpecRNET':
            loss = loss_spec + loss_spec_edge
            loss_ret = {'loss_spec': loss_spec, 'loss_spec_edge': loss_spec_edge}
        elif self.arch == 'SSRNET':
            loss = loss_fus + loss_spat_edge + loss_spec_edge
            loss_ret = {'loss_fus': loss_fus, 'loss_spat_edge': loss_spat_edge, 'loss_spec_edge': loss_spec_edge}

        return out, (loss, loss_ret)
    
    @torch.no_grad()
    def val_step(self, lrhsi, up_hs, rgb):
        out, out_spat, out_spec, edge_spat1, edge_spat2, edge_spec = self._forward_implem(up_hs, rgb)
        
        return out
    
# test
if __name__ == '__main__':
    net = SSRNET('SSRNET', 31, 3, 2)
    
    ms = torch.randn(1, 31, 16, 16)
    lms = torch.randn(1, 31, 32, 32)
    rgb = torch.randn(1, 3, 32, 32)
    
    print(
        net._forward_implem(lms, rgb)[0].shape
    )