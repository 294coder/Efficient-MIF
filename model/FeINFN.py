import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from model.module.fe_block import make_edsr_baseline, make_coord, ComplexGaborLayer, PositionalEmbedding, MLP_P, MLP, hightfre, ImplicitDecoder
from model.base_model import BaseModel, register_model
from model.module import PatchMergeModule
    
class FourierUnit(nn.Module):

    def __init__(self, feat_dim=128, guide_dim=128, mlp_dim=[256, 128], NIR_dim=33, d_model=2):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim

        imnet_in_dim = self.feat_dim + self.guide_dim + 2
        self.imnet1 = MLP(imnet_in_dim, out_dim=NIR_dim, hidden_list=self.mlp_dim)
        self.imnet2 = MLP_P(imnet_in_dim, out_dim=NIR_dim, hidden_list=self.mlp_dim)

        # self.pe = PositionalEmbedding(d_model, max_len=4096)

    def query_freq_a(self, feat, coord, hr_guide, mlp):
        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord # b x (64x64) x 2
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)

                pred = mlp(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)

        return ret
    def query_freq_p(self, feat, coord, hr_guide, mlp):
        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord # b x (64x64) x 2
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1).view(B, -1, H, W)

                pred = mlp(inp).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)
        return ret

    def forward(self, feat, coord, hr_guide):
        feat_ffted = torch.fft.fftn(feat, dim=(-2,-1))
        guide_ffted = torch.fft.fftn(hr_guide, dim=(-2,-1))

        feat_mag = torch.abs(feat_ffted)
        feat_pha = torch.angle(feat_ffted)
        guide_mag = torch.abs(guide_ffted)
        guide_pha = torch.angle(guide_ffted)

        ffted_mag = self.query_freq_a(feat_mag, coord, guide_mag, self.imnet1)
        ffted_pha = self.query_freq_p(feat_pha, coord, guide_pha, self.imnet2)

        real = ffted_mag * torch.cos(ffted_pha)
        imag = ffted_mag * torch.sin(ffted_pha)
        ffted = torch.complex(real, imag)

        output = torch.fft.ifftn(ffted, dim =(-2,-1))
        output = torch.abs(output)
        # output = output.real

        return output         

@register_model('FeINFN')
class FeINFNet(BaseModel):
    def __init__(self, hsi_dim=31, msi_dim=3,feat_dim=128, guide_dim=128, spa_edsr_num=4, spe_edsr_num=4, mlp_dim=[256, 128], NIR_dim=33, d_model=2,
                 scale=4, patch_merge=False,):
        super().__init__() 
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.NIR_dim = NIR_dim
        self.d_model = d_model
        self.scale = scale

        self.spatial_encoder = make_edsr_baseline(n_resblocks=spa_edsr_num, n_feats=self.guide_dim, n_colors=hsi_dim+msi_dim)
        self.spectral_encoder = make_edsr_baseline(n_resblocks=spe_edsr_num, n_feats=self.feat_dim, n_colors=hsi_dim)

        imnet_in_dim = self.feat_dim + self.guide_dim + self.feat_dim + 2
        self.imnet = MLP(imnet_in_dim, out_dim=NIR_dim, hidden_list=self.mlp_dim)
        self.hp = hightfre(in_channels=feat_dim, groups=1)
        self.decoder = ImplicitDecoder(in_channels=NIR_dim - 1, freq_dim=NIR_dim - 1, hidden_dims=[128, 128, 128],
                                       omega=30, scale=10.0)
        self.pe = PositionalEmbedding(d_model, max_len=4096)
        self.freq_query = FourierUnit(feat_dim, guide_dim, mlp_dim, NIR_dim)

        self.patch_merge = patch_merge
        self._patch_merge_model = PatchMergeModule(self, crop_batch_size=32,
                                                   scale=self.scale, 
                                                   patch_size_list=[16, 16*self.scale, 16*self.scale])

    def query(self, feat, coord, hr_guide):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry
                
                hp = self.hp(feat)

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                hp_feat = F.grid_sample(hp, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c])
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord 
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w
                rel_coord = self.pe(rel_coord)

                inp = torch.cat([q_feat, q_guide_hr, hp_feat, rel_coord], dim=-1)

                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def _forward_implem(self, HR_MSI, lms, LR_HSI):
        # HR_MSI Bx3x64x64
        # lms Bx31x64x64
        # LR_HSI Bx31x16x16

        _, _, H, W = HR_MSI.shape
        coord = make_coord([H, W]).cuda()
        feat = torch.cat([HR_MSI, lms], dim=1)
        hr_spa = self.spatial_encoder(feat)  # Bx128xHxW
        lr_spe = self.spectral_encoder(LR_HSI)  # Bx128xhxw The feature map of LR-HSI

        freq_feature = self.freq_query(lr_spe, coord, hr_spa)
        NIR_feature = self.query(lr_spe, coord, hr_spa) 

        output = self.decoder(NIR_feature, freq_feature) 
        output = lms + output

        return output

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self._forward_implem(pan, lms, ms)
        loss = criterion(sr, gt)
        
        return sr.clip(0, 1), loss

    def val_step(self, ms, lms, pan):        
        if self.patch_merge:
            pred = self._patch_merge_model.forward_chop(ms, lms, pan)[0]
        else:
            pred = self._forward_implem(pan, lms, ms)

        return pred.clip(0, 1)

    def patch_merge_step(self, ms, lms, pan, *args, **kwargs):
        return self._forward_implem(pan, lms, ms)


if __name__ == '__main__':
    torch.cuda.set_device('cuda:0')

    net = FeINFNet(hsi_dim=31, msi_dim=3,feat_dim=128, guide_dim=128, spa_edsr_num=4, spe_edsr_num=4, mlp_dim=[256, 128], NIR_dim=33).cuda()

    B, C, H, W = 1, 31, 64, 64
    scale = 4

    HR_MSI = torch.randn([B, 3, H, W]).cuda()
    lms = torch.randn([B, C, H, W]).cuda()
    LR_HSI = torch.randn([B, C, H // scale, W // scale]).cuda()
    gt = torch.randn(1, 31, H, W).cuda()
    criterion = torch.nn.L1Loss()
    output,loss = net.train_step(LR_HSI,lms,HR_MSI,gt,criterion)
    print(output.shape)
