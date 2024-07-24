import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from model.base_model import PatchMergeModule


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.E1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )

    def forward(self, x):
        E1 = self.E1(x)  # 512*512
        E2 = self.E2(F.avg_pool2d(E1, kernel_size=2, stride=2))  # 256*256
        E3 = self.E3(F.avg_pool2d(E2, kernel_size=2, stride=2))  # 128*128
        E4 = self.E4(F.avg_pool2d(E3, kernel_size=2, stride=2))  # 64*64
        E5 = self.E5(F.avg_pool2d(E4, kernel_size=2, stride=2))  # 32*32
        return E1, E2, E3, E4, E5


class Decoder(nn.Module):
    def __init__(self, Ch=31, kernel_size=[7, 7, 7]):
        super(Decoder, self).__init__()
        self.upMode = 'bicubic'
        self.Ch = Ch
        out_channel1 = Ch * kernel_size[0]
        out_channel2 = Ch * kernel_size[1]
        out_channel3 = Ch * kernel_size[2]
        self.D1 = nn.Sequential(nn.Conv2d(in_channels=512+512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D2 = nn.Sequential(nn.Conv2d(in_channels=512+256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D3 = nn.Sequential(nn.Conv2d(in_channels=256+128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D4 = nn.Sequential(nn.Conv2d(in_channels=128+64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.K_1      = nn.Sequential(nn.Conv2d(128, out_channel1, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channel1, out_channel1, 1, 1, 0)
                                      )
        self.K_2      = nn.Sequential(nn.Conv2d(128, out_channel2, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channel2, out_channel2, 1, 1, 0)
                                      )
        self.K_3      = nn.Sequential(nn.Conv2d(128, out_channel3, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channel3, out_channel3, 1, 1, 0)
                                      )

    def forward(self, E1, E2, E3, E4, E5):
        D1 = self.D1(torch.cat([E4, F.interpolate(E5, scale_factor=2, mode=self.upMode)], dim=1))
        D2 = self.D2(torch.cat([E3, F.interpolate(D1, scale_factor=2, mode=self.upMode)], dim=1))
        D3 = self.D3(torch.cat([E2, F.interpolate(D2, scale_factor=2, mode=self.upMode)], dim=1))
        D4 = self.D4(torch.cat([E1, F.interpolate(D3, scale_factor=2, mode=self.upMode)], dim=1))

        # filter generator
        k1 = self.K_1(D4)
        k2 = self.K_2(D4)
        k3 = self.K_3(D4)
        return k1, k2, k3


class HSI_Fusion(nn.Module):
    def __init__(self, Ch, stages, sf):
        super(HSI_Fusion, self).__init__()
        self.Ch = Ch
        self.s  = stages
        self.sf = sf
        self.kernel_size = [7, 7, 7]

        ## The modules for learning the measurement matrix R and R^T
        self.RT = nn.Sequential(nn.Conv2d(3, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
        self.R  = nn.Sequential(nn.Conv2d(Ch, 3, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())

        ## The modules for learning the measurement matrix B and B^T
        if self.sf == 8:
            self.BT = nn.Sequential(nn.ConvTranspose2d(Ch, Ch, kernel_size=12, stride=8, padding=2), nn.LeakyReLU())
            self.B  = nn.Sequential(nn.Conv2d(Ch, Ch, kernel_size=12, stride=8, padding=2), nn.LeakyReLU())
        if self.sf == 4:
            self.BT = nn.Sequential(nn.ConvTranspose2d(Ch, Ch, kernel_size=6, stride=4, padding=1), nn.LeakyReLU())
            self.B  = nn.Sequential(nn.Conv2d(Ch, Ch, kernel_size=6, stride=4, padding=1), nn.LeakyReLU())
        elif self.sf == 16:
            self.BT = nn.Sequential(nn.ConvTranspose2d(Ch, Ch, kernel_size=6, stride=4, padding=1),
                                    nn.LeakyReLU(),
                                    nn.ConvTranspose2d(Ch, Ch, kernel_size=6, stride=4, padding=1), nn.LeakyReLU())
            self.B = nn.Sequential(nn.Conv2d(Ch, Ch, kernel_size=6, stride=4, padding=1),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(Ch, Ch, kernel_size=6, stride=4, padding=1), nn.LeakyReLU())

        ## Encoding blocks
        self.Encoder = Encoder()

        ## Decoding blocks
        self.Decoder   = Decoder(Ch=self.Ch, kernel_size=self.kernel_size)

        self.conv  = nn.Conv2d(Ch+3, 64, kernel_size=3, stride=1, padding=1)

        ## Dense connection
        self.Den_con1 = nn.Conv2d(64    , 64, kernel_size=3, stride=1, padding=1)
        self.Den_con2 = nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1)
        self.Den_con3 = nn.Conv2d(64 * 3, 64, kernel_size=3, stride=1, padding=1)
        self.Den_con4 = nn.Conv2d(64 * 4, 64, kernel_size=3, stride=1, padding=1)
        # self.Den_con5 = nn.Conv2d(64 * 5, 64, kernel_size=1, stride=1, padding=0)
        # self.Den_con6 = nn.Conv2d(64 * 6, 64, kernel_size=1, stride=1, padding=0)


        self.lamda_0 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_0 = Parameter(torch.ones(1), requires_grad=True)
        self.lamda_1 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_1 = Parameter(torch.ones(1), requires_grad=True)
        self.lamda_2 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_2 = Parameter(torch.ones(1), requires_grad=True)
        self.lamda_3 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_3 = Parameter(torch.ones(1), requires_grad=True)
        # self.lamda_4 = Parameter(torch.ones(1), requires_grad=True)
        # self.eta_4 = Parameter(torch.ones(1), requires_grad=True)
        # self.lamda_5 = Parameter(torch.ones(1), requires_grad=True)
        # self.eta_5 = Parameter(torch.ones(1), requires_grad=True)

        self._initialize_weights()
        torch.nn.init.normal_(self.lamda_0, mean=1, std=0.01)
        torch.nn.init.normal_(self.eta_0, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lamda_1, mean=1, std=0.01)
        torch.nn.init.normal_(self.eta_1, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lamda_2, mean=1, std=0.01)
        torch.nn.init.normal_(self.eta_2, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lamda_3, mean=1, std=0.01)
        torch.nn.init.normal_(self.eta_3, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.lamda_4, mean=1, std=0.01)
        # torch.nn.init.normal_(self.eta_4, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.lamda_5, mean=1, std=0.01)
        # torch.nn.init.normal_(self.eta_5, mean=0.1, std=0.01)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def kernel_pred_1(self, cube, core):
        batch_size, bandwidth, height, width = cube.size()
        cube_pad = F.pad(cube, [self.kernel_size[0] // 2, self.kernel_size[0] // 2, 0, 0], mode='replicate')
        img_stack = []
        for i in range(self.kernel_size[0]):
            img_stack.append(cube_pad[:, :, :, i:i + width])
        img_stack = torch.stack(img_stack, dim=1)
        out_0 = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
        return out_0

    def kernel_pred_2(self, cube, core):
        batch_size, bandwidth, height, width = cube.size()
        cube_pad = F.pad(cube, [0, 0, self.kernel_size[1] // 2, self.kernel_size[1] // 2], mode='replicate')
        img_stack = []
        for i in range(self.kernel_size[1]):
            img_stack.append(cube_pad[:, :, i:i + height, :])
        img_stack = torch.stack(img_stack, dim=1)
        out_0 = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
        return out_0

    def kernel_pred_3(self, cube, core):
        batch_size, bandwidth, height, width = cube.size()
        cube_pad = F.pad(cube.unsqueeze(0).unsqueeze(0), pad=(0, 0, 0, 0, self.kernel_size[2] // 2, self.kernel_size[2] // 2)).squeeze(0).squeeze(0)
        img_stack = []
        for i in range(self.kernel_size[2]):
            img_stack.append(cube_pad[:, i:i + bandwidth, :, :])
        img_stack = torch.stack(img_stack, dim=1)
        out_0 = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
        return out_0

    def reconnect(self, Res1, Res2, Xt, Ut, i):
        if i == 0 :
            eta = self.eta_0
            lamda = self.lamda_0
        elif i == 1:
            eta = self.eta_1
            lamda = self.lamda_1
        elif i == 2:
            eta = self.eta_2
            lamda = self.lamda_2
        elif i == 3:
            eta = self.eta_3
            lamda = self.lamda_3
        # elif i == 4:
        #     eta = self.eta_4
        #     lamda = self.lamda_4
        # elif i == 5:
        #     eta = self.eta_5
        #     lamda = self.lamda_5

        Xt     =   Xt - 2 * eta * (Res1 + Res2  + lamda * (Xt - Ut))
        return Xt

    def _forward_implem(self, Y, X):
        re_list = []
        ## Initialize Z(0) with bicubic interpolation.
        Zt = F.interpolate(X, scale_factor=self.sf, mode='bicubic', align_corners=False)  # Z^(0)

        for i in range(0, self.s):
            ZtR = self.R(Zt)
            Res1 = self.RT(ZtR - Y)

            BZt = self.B(Zt)
            Res2 = self.BT(BZt - X)

            feat = self.conv(torch.cat((Zt, Y), 1))

            if i == 0:
                re_list.append(feat)
                fufeat = self.Den_con1(feat)
            elif i == 1:
                re_list.append(feat)
                fufeat = self.Den_con2(torch.cat(re_list, 1))
            elif i == 2:
                re_list.append(feat)
                fufeat = self.Den_con3(torch.cat(re_list, 1))
            elif i == 3:
                re_list.append(feat)
                fufeat = self.Den_con4(torch.cat(re_list, 1))
            # elif i == 4:
            #     re_list.append(fea)
            #     fufeat = self.Den_con5(torch.cat(re_list, 1))
            # elif i == 5:
            #     re_list.append(fea)
            #     fufeat = self.Den_con6(torch.cat(re_list, 1))

            ## Estimate the filtering matrix K
            E1, E2, E3, E4, E5 = self.Encoder(fufeat)
            k1, k2, k3 = self.Decoder(E1, E2, E3, E4, E5)

            batch_size, p, height, width = k1.size()
            k1                           = F.normalize(k1.view(batch_size, self.kernel_size[0], self.Ch, height, width), dim = 1)
            batch_size, p, height, width = k2.size()
            k2                           = F.normalize(k2.view(batch_size, self.kernel_size[1], self.Ch, height, width), dim = 1)
            batch_size, p, height, width = k3.size()
            k3                           = F.normalize(k3.view(batch_size, self.kernel_size[2], self.Ch, height, width), dim = 1)

            ## Compute the filtered result U(t) using K(t) and Z(t);
            Ut = self.kernel_pred_1(Zt, k1)
            Ut = self.kernel_pred_2(Ut, k2)
            Ut = self.kernel_pred_3(Ut, k3)

            ## Update Z(t) via Eq.(6)
            Zt = self.reconnect(Res1, Res2, Zt, Ut, i)
        return Zt
    
    def patch_merge_step(self, ms, lms, pan, *args, **kwargs):
        sr = self._forward_implem(pan, ms)
        
        return sr
    
if __name__ == "__main__":
    device = 'cuda:1'
    net = HSI_Fusion(31, 4, 8).to(device)
    ms = torch.randn(1, 31, 8, 8).to(device)
    hs = torch.randn(1, 3, 64, 64).to(device)
    
    with torch.no_grad():
        # import time
        # t1 = time.time()
        # for _ in range(10):
        sr = net._forward_implem(hs, ms)
        
        # net = PatchMergeModule(net, crop_batch_size=12, patch_size_list=[16, 64, 64])
        
        # sr = net.forward_chop(ms, hs, hs)[0]
        # import time
        # t1 = time.time()
        # for _ in range(10):
        #     sr = net.forward_chop(ms, hs, hs)[0]
            
        # print(sr.shape)
        # print(f'===> test time: {(time.time()-t1)/10}')
        
        from fvcore.nn import flop_count_table, FlopCountAnalysis, parameter_count_table

        net.forward = net._forward_implem
        flops = FlopCountAnalysis(net, (hs, ms))
        print(flop_count_table(flops, max_depth=3))
        print(flops.total() / 1e9, 'G')
