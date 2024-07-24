from functools import partial
import random
from typing import Sequence, Union
from einops import reduce
from contextlib import contextmanager
import kornia
from kornia.filters import spatial_gradient
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import numpy as np
from math import exp
import lpips
from deepinv.loss import TVLoss

import sys

sys.path.append('./')
from model.base_model import BaseModel
from utils.misc import is_main_process, exists, default, rgb_to_ycbcr, ycbcr_to_rgb
from utils.torch_dct import dct_2d, idct_2d
from utils.vgg import vgg16
from utils._ydtr_loss import ssim_loss_ir, ssim_loss_vi, sf_loss_ir, sf_loss_vi
from utils.log_utils import easy_logger

logger = easy_logger()


class PerceptualLoss(nn.Module):
    def __init__(self, percep_net="vgg", norm=True):
        super(PerceptualLoss, self).__init__()
        self.norm = norm
        self.lpips_loss = lpips.LPIPS(net=percep_net).cuda()

    def forward(self, x, y):
        # assert x.shape == y.shape
        loss = self.lpips_loss(x, y, normalize=self.norm)
        return torch.squeeze(loss).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


class MaxGradientLoss(torch.nn.Module):
    def __init__(self, mean_batch=True) -> None:
        super().__init__()
        self.register_buffer(
            "x_sobel_kernel",
            torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).expand(1, 1, 3, 3),
        )
        self.register_buffer(
            "y_sobel_kernel",
            torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).expand(1, 1, 3, 3),
        )
        self.mean_batch = mean_batch

    def forward(self, fuse, ir, vis):
        c = fuse.size(1)

        fuse_grad_x = F.conv2d(fuse, self.x_sobel_kernel, padding=1, groups=c)
        fuse_grad_y = F.conv2d(fuse, self.y_sobel_kernel, padding=1, groups=c)

        ir_grad_x = F.conv2d(ir, self.x_sobel_kernel, padding=1, groups=c)
        ir_grad_y = F.conv2d(ir, self.y_sobel_kernel, padding=1, groups=c)

        vis_grad_x = F.conv2d(vis, self.x_sobel_kernel, padding=1, groups=c)
        vis_grad_y = F.conv2d(vis, self.y_sobel_kernel, padding=1, groups=c)

        max_grad_x = torch.maximum(ir_grad_x, vis_grad_x)
        max_grad_y = torch.maximum(ir_grad_y, vis_grad_y)

        if self.mean_batch:
            max_gradient_loss = (
                F.l1_loss(fuse_grad_x, max_grad_x) + F.l1_loss(fuse_grad_y, max_grad_y)
            ) / 2
        else:
            x_loss_b = F.l1_loss(fuse_grad_x, max_grad_x, reduction="none").mean(
                dim=(1, 2, 3)
            )
            y_loss_b = F.l1_loss(fuse_grad_y, max_grad_y, reduction="none").mean(
                dim=(1, 2, 3)
            )

            max_gradient_loss = (x_loss_b + y_loss_b) / 2

        return max_gradient_loss


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def mci_loss(pred, gt):
    return F.l1_loss(pred, gt.max(1, keepdim=True)[0])


def sf(f1, kernel_radius=5):
    """copy from https://github.com/tthinking/YDTR/blob/main/losses/__init__.py

    Args:
        f1 (torch.Tensor): image shape [b, c, h, w]
        kernel_radius (int, optional): kernel redius using calculate sf. Defaults to 5.

    Returns:
        loss: loss item. type torch.Tensor
    """

    device = f1.device
    b, c, h, w = f1.shape
    r_shift_kernel = (
        torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        .to(device)
        .reshape((1, 1, 3, 3))
        .repeat(c, 1, 1, 1)
    )
    b_shift_kernel = (
        torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        .to(device)
        .reshape((1, 1, 3, 3))
        .repeat(c, 1, 1, 1)
    )
    f1_r_shift = F.conv2d(f1, r_shift_kernel, padding=1, groups=c)
    f1_b_shift = F.conv2d(f1, b_shift_kernel, padding=1, groups=c)
    f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
    kernel_size = kernel_radius * 2 + 1
    add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().to(device)
    kernel_padding = kernel_size // 2
    f1_sf = torch.sum(
        F.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1
    )
    return 1 - f1_sf


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class HybridL1L2(torch.nn.Module):
    def __init__(self):
        super(HybridL1L2, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
        self.loss = LossWarpper(l1=self.l1, l2=self.l2)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss, loss_dict


class HybridSSIMSF(torch.nn.Module):
    def __init__(self, channel, weighted_r=(1.0, 5e-2, 6e-4, 25e-5)) -> None:
        super().__init__()
        self.weighted_r = weighted_r

    def forward(self, fuse, gt):
        # fuse: [b, 1, h, w]
        vi = gt[:, 0:1]  # [b, 1, h, w]
        ir = gt[:, 1:]  # [b, 1, h, w]

        _ssim_f_ir = ssim_loss_ir(fuse, ir)
        _ssim_f_vi = ssim_loss_vi(fuse, vi)
        _sf_f_ir = sf_loss_ir(fuse, ir)
        _sf_f_vi = sf_loss_vi(fuse, vi)

        ssim_f_ir = self.weighted_r[0] * _ssim_f_ir
        ssim_f_vi = self.weighted_r[1] * _ssim_f_vi
        sf_f_ir = self.weighted_r[2] * _sf_f_ir
        sf_f_vi = self.weighted_r[3] * _sf_f_vi

        loss_dict = dict(
            ssim_f_ir=ssim_f_ir,
            ssim_f_vi=ssim_f_vi,
            sf_f_ir=sf_f_ir,
            sf_f_vi=sf_f_vi,
        )

        loss = ssim_f_ir + ssim_f_vi + sf_f_ir + sf_f_vi
        return loss, loss_dict


class HybridSSIMMCI(torch.nn.Module):
    def __init__(self, channel, weight_r=(1.0, 1.0, 1.0)) -> None:
        super().__init__()
        self.ssim = SSIMLoss(channel=channel)
        self.mci_loss = mci_loss
        self.weight_r = weight_r

    def forward(self, fuse, gt):
        # fuse: [b, 1, h, w]
        vi = gt[:, 0:1]  # [b, 1, h, w]
        ir = gt[:, 1:]  # [b, 1, h, w]

        _ssim_f_ir = self.weight_r[0] * self.ssim(fuse, ir)
        _ssim_f_vi = self.weight_r[1] * self.ssim(fuse, vi)
        _mci_loss = self.weight_r[2] * self.mci_loss(fuse, gt)

        loss = _ssim_f_ir + _ssim_f_vi + _mci_loss

        loss_dict = dict(
            ssim_f_ir=_ssim_f_ir,
            ssim_f_vi=_ssim_f_vi,
            mci_loss=_mci_loss,
        )

        return loss, loss_dict


def accum_loss_dict(ep_loss_dict: dict, loss_dict: dict):
    for k, v in loss_dict.items():
        if k in ep_loss_dict:
            ep_loss_dict[k] += v
        else:
            ep_loss_dict[k] = v
    return ep_loss_dict


def ave_ep_loss(ep_loss_dict: dict, ep_iters: int):
    for k, v in ep_loss_dict.items():
        ep_loss_dict[k] = v / ep_iters
    return ep_loss_dict

@is_main_process  
def ave_multi_rank_dict(rank_loss_dict: list[dict]):
    ave_dict = {}
    n = len(rank_loss_dict)
    assert n >= 1, "@rank_loss_dict must have at least one element"
    keys = rank_loss_dict[0].keys()

    for k in keys:
        vs = 0
        for d in rank_loss_dict:
            v = d[k]
            vs = vs + v
        ave_dict[k] = vs / n
    return ave_dict

class HybridL1SSIM(torch.nn.Module):
    def __init__(self, channel=31, weighted_r=(1.0, 0.1)):
        super(HybridL1SSIM, self).__init__()
        assert len(weighted_r) == 2
        self._l1 = torch.nn.L1Loss()
        self._ssim = SSIMLoss(channel=channel)
        self.loss = LossWarpper(weighted_r, l1=self._l1, ssim=self._ssim)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss, loss_dict


class HybridCharbonnierSSIM(torch.nn.Module):
    def __init__(self, weighted_r, channel=31) -> None:
        super().__init__()
        self._ssim = SSIMLoss(channel=channel)
        self._charb = CharbonnierLoss(eps=1e-4)
        self.loss = LossWarpper(weighted_r, charbonnier=self._charb, ssim=self._ssim)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return (loss,)


class HybridMCGMCI(torch.nn.Module):
    def __init__(self, weight_r=(1.0, 1.0)) -> None:
        super().__init__()
        self.mcg = MaxGradientLoss()
        self.mci = mci_loss
        self.weight_r = weight_r

    def forward(self, pred, gt):
        vis = gt[:, 0:1]
        ir = gt[:, 1:]

        mcg_loss = self.mcg(pred, ir, vis) * self.weight_r[0]
        mci_loss = self.mci(pred, gt) * self.weight_r[1]

        loss_dict = dict(mcg=mcg_loss, mci=mci_loss)

        return mcg_loss + mci_loss, loss_dict


def gradient(input):
    """
    求图像梯度, sobel算子
    :param input:
    :return:
    """

    filter1 = nn.Conv2d(
        kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1
    )
    filter2 = nn.Conv2d(
        kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1
    )
    filter1.weight.data = (
        torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        .reshape(1, 1, 3, 3)
        .to(input.device)
    )
    filter2.weight.data = (
        torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        .reshape(1, 1, 3, 3)
        .to(input.device)
    )

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient


class LossWarpper(torch.nn.Module):
    def __init__(self, weighted_ratio=(1.0, 1.0), **losses):
        super(LossWarpper, self).__init__()
        self.names = []
        assert len(weighted_ratio) == len(losses.keys())
        self.weighted_ratio = weighted_ratio
        for k, v in losses.items():
            self.names.append(k)
            setattr(self, k, v)

    def forward(self, pred, gt) -> tuple[torch.Tensor, dict[torch.Tensor]]:
        loss = 0.0
        d_loss = {}
        for i, n in enumerate(self.names):
            l = getattr(self, n)(pred, gt) * self.weighted_ratio[i]
            loss += l
            d_loss[n] = l
        return loss, d_loss


class TorchLossWrapper(torch.nn.Module):
    def __init__(self, weight_ratio: Union[tuple[float], list[float]], **loss) -> None:
        super().__init__()
        self.key = list(loss.keys())
        self.loss = list(loss.values())
        self.weight_ratio = weight_ratio

        assert len(weight_ratio) == len(loss.keys())

    def forward(self, pred, gt):
        loss_total = 0.0
        loss_d = {}
        for i, l in enumerate(self.loss):
            loss_i = l(pred, gt) * self.weight_ratio[i]
            loss_total = loss_total + loss_i

            k = self.key[i]
            loss_d[k] = loss_i

        return loss_total, loss_d


class SSIMLoss(torch.nn.Module):
    def __init__(
        self, win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3
    ):
        super(SSIMLoss, self).__init__()
        self.window_size = win_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(win_size, self.channel, win_sigma)
        self.win_sigma = win_sigma

    def forward(self, img1, img2):
        # print(img1.size())
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel, self.win_sigma)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - _ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )


def ssim(img1, img2, win_size=11, data_range=1, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(win_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, win_size, channel, size_average)


def elementwise_charbonnier_loss(
    input: Tensor, target: Tensor, eps: float = 1e-3
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    return torch.sqrt((input - target) ** 2 + (eps * eps))


class HybridL1L2(nn.Module):
    def __init__(self, cof=10.0):
        super(HybridL1L2, self).__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.cof = cof

    def forward(self, pred, gt):
        return self.l1(pred, gt) / self.cof + self.l2(pred, gt)


class RMILoss(nn.Module):
    def __init__(
        self,
        with_logits=False,
        radius=3,
        bce_weight=0.5,
        downsampling_method="max",
        stride=3,
        use_log_trace=True,
        use_double_precision=True,
        epsilon=0.0005,
    ):

        super().__init__()

        self.use_double_precision = use_double_precision
        self.with_logits = with_logits
        self.bce_weight = bce_weight
        self.stride = stride
        self.downsampling_method = downsampling_method
        self.radius = radius
        self.use_log_trace = use_log_trace
        self.epsilon = epsilon

    def forward(self, input, target):

        if self.bce_weight != 0:
            if self.with_logits:
                bce = F.binary_cross_entropy_with_logits(input, target=target)
            else:
                bce = F.binary_cross_entropy(input, target=target)
            bce = bce.mean() * self.bce_weight
        else:
            bce = 0.0

        if self.with_logits:
            input = torch.sigmoid(input)

        rmi = self.rmi_loss(input=input, target=target)
        rmi = rmi.mean() * (1.0 - self.bce_weight)
        return rmi + bce

        return bce

    def rmi_loss(self, input, target):

        assert input.shape == target.shape
        vector_size = self.radius * self.radius

        y = self.extract_region_vector(target)
        p = self.extract_region_vector(input)

        if self.use_double_precision:
            y = y.double()
            p = p.double()

        eps = torch.eye(vector_size, dtype=y.dtype, device=y.device) * self.epsilon
        eps = eps.unsqueeze(dim=0).unsqueeze(dim=0)

        y = y - y.mean(dim=3, keepdim=True)
        p = p - p.mean(dim=3, keepdim=True)

        y_cov = y @ self.transpose(y)
        p_cov = p @ self.transpose(p)
        y_p_cov = y @ self.transpose(p)

        m = y_cov - y_p_cov @ self.transpose(
            self.inverse(p_cov + eps)
        ) @ self.transpose(y_p_cov)

        if self.use_log_trace:
            rmi = 0.5 * self.log_trace(m + eps)
        else:
            rmi = 0.5 * self.log_det(m + eps)

        rmi = rmi / float(vector_size)

        return rmi.sum(dim=1).mean(dim=0)

    def extract_region_vector(self, x):
        x = self.downsample(x)
        stride = self.stride if self.downsampling_method == "region-extraction" else 1

        x_regions = F.unfold(x, kernel_size=self.radius, stride=stride)
        x_regions = x_regions.view((*x.shape[:2], self.radius**2, -1))
        return x_regions

    def downsample(self, x):

        if self.stride == 1:
            return x

        if self.downsampling_method == "region-extraction":
            return x

        padding = self.stride // 2
        if self.downsampling_method == "max":
            return F.max_pool2d(
                x, kernel_size=self.stride, stride=self.stride, padding=padding
            )
        if self.downsampling_method == "avg":
            return F.avg_pool2d(
                x, kernel_size=self.stride, stride=self.stride, padding=padding
            )
        raise ValueError(self.downsampling_method)

    @staticmethod
    def transpose(x):
        return x.transpose(-2, -1)

    @staticmethod
    def inverse(x):
        return torch.inverse(x)

    @staticmethod
    def log_trace(x):
        x = torch.linalg.cholesky(x)
        diag = torch.diagonal(x, dim1=-2, dim2=-1)
        return 2 * torch.sum(torch.log(diag + 1e-8), dim=-1)

    @staticmethod
    def log_det(x):
        return torch.logdet(x)


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, img1, img2) -> Tensor:
        return elementwise_charbonnier_loss(img1, img2, eps=self.eps).mean()


class HybridSSIMRMIFuse(nn.Module):
    def __init__(self, weight_ratio=(1.0, 1.0), ssim_channel=1):
        super().__init__()
        # self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(channel=ssim_channel)
        self.rmi = RMILoss(bce_weight=0.6)
        self.weight_ratio = weight_ratio

    def forward(self, fuse, x):
        fuse = fuse.clip(0, 1)

        vis = x[:, 0:1]
        ir = x[:, 1:]

        ssim_loss = self.ssim(fuse, vis) + self.ssim(fuse, ir)
        rmi_loss = self.rmi(fuse, vis) + self.rmi(fuse, ir)

        loss_d = dict(ssim=ssim_loss, rmi=rmi_loss)

        loss = self.weight_ratio[0] * ssim_loss + self.weight_ratio[1] * rmi_loss
        return loss, loss_d


class HybridPIALoss(nn.Module):
    def __init__(self, weight_ratio=(3, 7, 20, 10)) -> None:
        super().__init__()
        assert (
            len(weight_ratio) == 4
        ), "@weight_ratio must be a tuple or list of length 4"
        self.weight_ratio = weight_ratio
        self._mcg_loss = MaxGradientLoss()
        self.perceptual_loss = PerceptualLoss(norm=True)

    def forward(self, fuse, gt):
        vis = gt[:, 0:1]
        ir = gt[:, 1:]

        l1_int = (F.l1_loss(fuse, vis) + F.l1_loss(fuse, ir)) * self.weight_ratio[0]
        l1_aux = (F.l1_loss(fuse, gt.max(1, keepdim=True)[0])) * self.weight_ratio[1]

        # FIXME: this should implement as the largest gradient of vis and ir
        # l1_grad = (F.l1_loss(gradient(fuse), gradient(vis)) + F.l1_loss(gradient(fuse), gradient(ir))) * \
        #           self.weight_ratio[2]
        l1_grad = self._mcg_loss(fuse, ir, vis) * self.weight_ratio[2]
        percep_loss = (
            self.perceptual_loss(fuse, vis) + self.perceptual_loss(fuse, ir)
        ) * self.weight_ratio[3]

        loss_d = dict(
            intensity_loss=l1_int,
            context_loss=l1_aux,
            gradient_loss=l1_grad,
            percep_loss=percep_loss,
        )

        return l1_int + l1_aux + l1_grad + percep_loss, loss_d


def parse_fusion_gt(gt: "Tensor | tuple[Tensor] | list[Tensor]"):
    # TODO: consider the vis is RGB
    if isinstance(gt, Tensor):
        if gt.size(1) == 4:
            ir, vi = gt[:, 3:], gt[:, :3]
        elif gt.size(1) == 2:
            ir, vi = gt[:, 1:], gt[:, 0:1]
    elif isinstance(gt, (tuple, list)):
        ir, vi = gt[1], gt[0]
    else:
        raise ValueError('gt must be a tensor or a tuple or a list')
    
    return ir, vi

# U2Fusion dynamic loss weight
class U2FusionLoss(nn.Module):
    def __init__(self, 
                 loss_weights: tuple[float, float, float] = (5., 2., 10.)) -> None:
        # loss_weights:
        super().__init__()
        # modified from https://github.com/ytZhang99/U2Fusion-pytorch/blob/master/train.py
        # and https://github.com/linklist2/PIAFusion_pytorch/blob/master/train_fusion_model.py
        # no normalization
        # so do not unormalize the input

        assert len(loss_weights) == 3, "loss_weights must be a tuple of length 3"

        self.feature_model = vgg16(pretrained=True)
        self.c = 0.1
        self.loss_weights = loss_weights
        self.ssim_loss = SSIMLoss(channel=1)
        #   , size_average=False)
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, fuse, gt, *, mask=None):
        
        # similiar to PIAFusion paper, which introduces a classifier
        # to judge the day or night image and give the probability
        ir, vi = parse_fusion_gt(gt)
        ir, vi = self.repeat_dims(ir), self.repeat_dims(vi)
        
        ws = self.dynamic_weight(ir, vi)
        ir_w, vi_w = ws.chunk(2, dim=-1)
        ir_w, vi_w = ir_w.flatten(), vi_w.flatten()

        # here we do not follow U2Fusion paper and change it into other losses
        l1_int = (
            vi_w * self.mse_loss(fuse, vi).mean((1, 2, 3))
            + ir_w * self.mse_loss(fuse, ir).mean((1, 2, 3))
        ).mean() * self.loss_weights[0]

        l1_aux = F.mse_loss(fuse, torch.max(ir, vi)) * self.loss_weights[1]

        # gradient part. choose the largest gradient
        # l1_grad = (
        #         F.l1_loss(
        #             gradient(fuse),
        #             torch.maximum(
        #                 vi_w[:, None, None, None] * gradient(gt[:, 0:1]),
        #                 ir_w[:, None, None, None] * gradient(gt[:, 1:]),
        #             ),
        #         )
        #         * self.loss_weights[2]
        # )

        # l1_grad = (
        #     self.l1_loss(gradient(fuse), vi_w * gradient(gt[:, 0:1])).mean((1, 2, 3))
        #     + self.l1_loss(gradient(fuse), ir_w * gradient(gt[:, 1:])).mean((1, 2, 3))
        # ).mean()

        # ssim loss would cause window artifacts
        # loss_ssim = (
        #     ir_w * self.ssim_loss(fuse, gt[:, 1:])
        #     + vi_w * self.ssim_loss(fuse, gt[:, 0:1])
        # ).mean() * self.loss_weights[2]
        loss_ssim = (
            self.ssim_loss(fuse, ir) + self.ssim_loss(fuse, vi)
        ) * self.loss_weights[2]

        loss_d = dict(intensity_loss=l1_int, aux_loss=l1_aux, ssim_loss=loss_ssim)
        # print(ir_w, vi_w)

        return l1_int + l1_aux + loss_ssim, loss_d

    @torch.no_grad()
    def dynamic_weight(self, ir_vgg, vi_vgg):

        ir_f = self.feature_model(ir_vgg)
        vi_f = self.feature_model(vi_vgg)

        m1s = []
        m2s = []
        for i in range(len(ir_f)):
            m1 = torch.mean(self.features_grad(ir_f[i]).pow(2), dim=[1, 2, 3])
            m2 = torch.mean(self.features_grad(vi_f[i]).pow(2), dim=[1, 2, 3])

            m1s.append(m1)
            m2s.append(m2)
            # if i == 0:
            #     w1 = torch.unsqueeze(m1, dim=-1)
            #     w2 = torch.unsqueeze(m2, dim=-1)
            # else:
            #     w1 = torch.cat((w1, torch.unsqueeze(m1, dim=-1)), dim=-1)
            #     w2 = torch.cat((w2, torch.unsqueeze(m2, dim=-1)), dim=-1)

        w1 = torch.stack(m1s, dim=-1)
        w2 = torch.stack(m2s, dim=-1)

        weight_1 = (torch.mean(w1, dim=-1) / self.c).detach()
        weight_2 = (torch.mean(w2, dim=-1) / self.c).detach()

        # print(weight_1.tolist()[:6], weight_2.tolist()[:6])

        weight_list = torch.stack(
            [weight_1, weight_2], dim=-1
        )  # torch.cat((weight_1.unsqueeze(-1), weight_2.unsqueeze(-1)), -1)
        weight_list = F.softmax(weight_list, dim=-1)

        return weight_list

    @staticmethod
    def features_grad(features):
        kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
        kernel = (
            torch.FloatTensor(kernel)
            .expand(features.shape[1], 1, 3, 3)
            .to(features.device)
        )
        feat_grads = F.conv2d(
            features, kernel, stride=1, padding=1, groups=features.shape[1]
        )
        # _, c, _, _ = features.shape
        # c = int(c)
        # for i in range(c):
        #     feat_grad = F.conv2d(
        #         features[:, i : i + 1, :, :], kernel, stride=1, padding=1
        #     )
        #     if i == 0:
        #         feat_grads = feat_grad
        #     else:
        #         feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
        return feat_grads

    def repeat_dims(self, x):
        assert x.size(1) in [1, 3], "the number of channel of x must be 3"
        if x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        else:
            return x


# DCT Blur Loss
class DCTBlurLoss(nn.Module):
    def __init__(self, temperature=100, reduction="mean") -> None:
        super().__init__()
        self.t = temperature
        self.reduction = reduction
        self.distance = nn.L1Loss(reduction=reduction)
        if reduction == "none":
            self.feature_model = vgg16(pretrained=True)
            self.c = 0.1

    @staticmethod
    def heat_blur_torch(img, t=25):
        K1 = img.shape[-2]
        K2 = img.shape[-1]

        dct_img = dct_2d(img, norm="ortho")  # [3, K1, K2]
        freqs_h = torch.pi * torch.linspace(0, K1 - 1, K1) / K1  # [K1]
        freqs_w = torch.pi * torch.linspace(0, K2 - 1, K2) / K2  # [K2]

        freq_square = (freqs_h[:, None] ** 2 + freqs_w[None, :] ** 2).to(
            img.device
        )  # [K1, K2]
        dct_img = dct_img * torch.exp(-freq_square[None, ...] * t)  # [3, K1, K2]

        recon_img = idct_2d(dct_img, norm="ortho")

        return recon_img

    def forward(self, f, gt):
        ws = self.dynamic_weight(gt)
        ir_w, vi_w = ws.chunk(2, dim=-1)
        ir_w, vi_w = ir_w.flatten(), vi_w.flatten()

        f_dct_blur = self.heat_blur_torch(f, t=self.t)
        vi_dct_blur, ir_dct_blur = self.heat_blur_torch(gt, t=self.t).chunk(2, dim=1)

        f_vi_loss = self.distance(f_dct_blur, vi_dct_blur)
        f_ir_loss = self.distance(f_dct_blur, ir_dct_blur)

        if self.reduction == "none":
            f_vi_loss = f_vi_loss.mean(dim=(1, 2, 3))
            f_ir_loss = f_ir_loss.mean(dim=(1, 2, 3))

            ws = self.dynamic_weight(gt)
            ir_w, vi_w = ws.chunk(2, dim=-1)
            ir_w, vi_w = ir_w.flatten(), vi_w.flatten()

            f_vi_loss = f_vi_loss * vi_w
            f_ir_loss = f_ir_loss * ir_w

        return (f_vi_loss + f_ir_loss).mean()

    @torch.no_grad()
    def dynamic_weight(self, gt):
        ir_vgg, vi_vgg = self.repeat_dims(gt[:, 1:]), self.repeat_dims(gt[:, 0:1])

        ir_f = self.feature_model(ir_vgg)
        vi_f = self.feature_model(vi_vgg)

        m1s = []
        m2s = []
        for i in range(len(ir_f)):
            m1 = torch.mean(self.features_grad(ir_f[i]).pow(2), dim=[1, 2, 3])
            m2 = torch.mean(self.features_grad(vi_f[i]).pow(2), dim=[1, 2, 3])

            m1s.append(m1)
            m2s.append(m2)

        w1 = torch.stack(m1s, dim=-1)
        w2 = torch.stack(m2s, dim=-1)

        weight_1 = torch.mean(w1, dim=-1) / self.c
        weight_2 = torch.mean(w2, dim=-1) / self.c
        weight_list = torch.stack([weight_1, weight_2], dim=-1)
        weight_list = F.softmax(weight_list, dim=-1)

        return weight_list

    @staticmethod
    def features_grad(features):
        kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
        kernel = (
            torch.FloatTensor(kernel)
            .expand(features.shape[1], 1, 3, 3)
            .to(features.device)
        )
        feat_grads = F.conv2d(
            features, kernel, stride=1, padding=1, groups=features.shape[1]
        )
        return feat_grads

    def repeat_dims(self, x):
        assert x.size(1) == 1, "the number of channel of x must be 1"
        return x.repeat(1, 3, 1, 1)

################# SwinFusion loss helper functions #################

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        image_fused_Y = image_fused[:, :1, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        # gradient_A = TF.gaussian_blur(gradient_A, 3, [1, 1])
        gradient_B = self.sobelconv(image_B_Y)
        # gradient_B = TF.gaussian_blur(gradient_B, 3, [1, 1])
        gradient_fused = self.sobelconv(image_fused_Y)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        weight_A = 0.5
        weight_B = 0.5
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM
class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        image_A = image_A.unsqueeze(0)
        image_B = image_B.unsqueeze(0)      
        intensity_joint = torch.mean(torch.cat([image_A, image_B]), dim=0)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity
    
################# SwinFusion loss #################

class SwinFusionLoss(nn.Module):
    def __init__(self):
        super(SwinFusionLoss, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        
    def forward(self, image_A, image_B, image_fused):
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        
        loss_d = {'loss_l1': loss_l1, 'loss_gradient': loss_gradient, 'loss_SSIM': loss_SSIM}
        return fusion_loss, loss_d #loss_gradient, loss_l1, loss_SSIM

#####################################################


class CDDFusionLoss(nn.Module):
    def __init__(self, weights=(1, 1, 1)) -> None:
        super().__init__()
        self.weights = weights

        self.l1_loss = nn.L1Loss()
        self.dct_loss = DCTBlurLoss(reduction="none")
        self.mcg_loss = MaxGradientLoss()

    def forward(self, f, gt):
        l1_loss = self.l1_loss(f, gt.max(dim=1, keepdim=True)[0]) * self.weights[0]
        dct_loss = self.dct_loss(f, gt) * self.weights[1]
        mcg_loss = self.mcg_loss(f, gt[:, 0:1], gt[:, 1:]) * self.weights[2]

        loss_d = dict(l1_loss=l1_loss, dct_loss=dct_loss, mcg_loss=mcg_loss)

        return l1_loss + dct_loss + mcg_loss, loss_d



### psfusion loss

class CorrelationLoss(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self, eps=1e-6):
        super(CorrelationLoss, self).__init__()
        self.eps = eps

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()
        r = torch.sum(img1*img2)/torch.sqrt(torch.sum(img1*img1)*torch.sum(img2*img2))
        return r

    def forward(self, image_ir, img_vis, img_fusion):
        cc = self.corr2(image_ir, img_fusion) + self.corr2(img_vis, img_fusion)
        
        return 1. / (cc + self.eps)


### DRMF loss ###

def RGB2YCrCb(rgb_image):
    """
    Convert RGB format to YCrCb format.
    Used in the intermediate results of the color space conversion, because the default size of rgb_image is [B, C, H, W].
    :param rgb_image: image data in RGB format
    :return: Y, Cr, Cb
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0)#.detach()
    Cb = Cb.clamp(0.0,1.0)#.detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    """
    Convert YcrCb format to RGB format
    :param Y.
    :param Cb.
    :param Cr.
    :return.
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out


class SobelOp(nn.Module):
    def __init__(self):
        super(SobelOp, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)
        
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        
        # sobel_xy = torch.abs(sobelx)+torch.abs(sobely)
        
        sobel_xy = torch.max(
            torch.abs(sobelx), torch.abs(sobely)
        )
        
        return sobel_xy
        

class DRMFFusionLoss(nn.Module):
    def __init__(self, 
                 latent_weighted: bool=False, 
                 *,
                 grad_loss: bool=True,
                 ssim_loss: bool=True,
                 tv_loss: bool=False, 
                 pseudo_l1_const=1.4e-3,
                 correlation_loss: bool=False,
                 reduce_label: bool=True,
                 color_loss_bg_masked: bool=False,
                 weight_dict: dict=None):
        super(DRMFFusionLoss, self).__init__()
        self.latent_weighted = latent_weighted
        self.ssim = ssim_loss
        self.grad = grad_loss        
        self.tv = tv_loss
        self.correlation = correlation_loss
        self.reduce_label = reduce_label
        self.color_loss_bg_masked = color_loss_bg_masked
        logger.info(f'{__class__.__name__}: color loss performs [green]{"only on background" if color_loss_bg_masked else "on the whole image"}[/green]')
        
        self.loss_func = nn.L1Loss(reduction='none') if pseudo_l1_const == 0 else \
                         partial(self.pseudo_l2_loss, c=pseudo_l1_const)
        main_loss = 'l1 loss' if pseudo_l1_const == 0 else 'pseudo l2 loss'
        
        if grad_loss:
            self.sobelconv = SobelOp()
        if ssim_loss:
            self.ssim_func = SSIMLoss()
        if tv_loss:
            self.tv_loss = TVLoss(weight=2.0)
        if correlation_loss:
            self.cc_loss = CorrelationLoss()
        
        if latent_weighted:
            logger.info('using vgg16 to extract latent features')
            self.latent_model = vgg16(pretrained=True).eval()
            self.latent_temp = 0.1
            feature_grad_kernel = torch.tensor([[1 / 8, 1 / 8, 1 / 8], 
                                                [1 / 8, -1, 1 / 8], 
                                                [1 / 8, 1 / 8, 1 / 8]]).type(torch.float32)
            self.register_buffer('kernel', feature_grad_kernel)

        self.weight_dict = default(weight_dict, {
                                                    'fusion_gt': 1,
                                                    'inten_f_joint': 2,
                                                    'inten_f_ir': 4,
                                                    'inten_f_vi': 2,
                                                    'color_f_cb': 2,
                                                    'color_f_cr': 2,
                                                    'grad_f_joint': 4,
                                                    'ssim_f_joint': 0.6,
                                                    'tv_f': 0.1,
                                                    'crr_f': 0.02,
                                                })
        
        logger.info(f'Latent weighted: {latent_weighted}, TV loss: {tv_loss}, grad loss: {grad_loss}, SSIM loss: {ssim_loss}, correlation loss: {correlation_loss}')
        logger.info(f'main loss: {main_loss}')
    
    @staticmethod
    def pseudo_l2_loss(img1, img2, c):
        return torch.sqrt((img1 - img2) ** 2 + c ** 2) - c
        
    def check_rgb(self, img):
        if img.size(1) != 3:
            assert img.size(1) == 1, 'The channel of the image should be 1 or 3.'
            img = img.repeat(1, 3, 1, 1)
            
        return img.detach()
    
    @torch.no_grad()
    def dynamic_weight(self, ir, vi):
        def features_grad(features):
            kernel = self.kernel.expand(features.shape[1], 1, 3, 3)
            feat_grads = F.conv2d(features, kernel, stride=1, padding=1, groups=features.shape[1])
            return feat_grads

        ir_f = self.latent_model(ir)
        vi_f = self.latent_model(vi)

        m1s = []
        m2s = []
        for i in range(len(ir_f)):
            m1 = torch.mean(features_grad(ir_f[i]).pow(2), dim=[1, 2, 3])
            m2 = torch.mean(features_grad(vi_f[i]).pow(2), dim=[1, 2, 3])

            m1s.append(m1)
            m2s.append(m2)

        w1 = torch.stack(m1s, dim=-1)
        w2 = torch.stack(m2s, dim=-1)

        weight_1 = torch.mean(w1, dim=-1) / self.latent_temp
        weight_2 = torch.mean(w2, dim=-1) / self.latent_temp

        weight_list = torch.stack([weight_1, weight_2], dim=-1)
        weight_list = F.softmax(weight_list, dim=-1)
        ir_w, vi_w = weight_list.chunk(2, dim=-1)
        ir_w, vi_w = ir_w.flatten(), vi_w.flatten()
        
        return vi_w, ir_w
    
    def check_dtype_and_device(self, *args: tuple[Tensor]):
        dtype = None
        device = None
        
        def _asserts(ti, dtype, device):
            assert ti.dtype == dtype, f'The dtype of the input tensors should be the same, but got {ti.dtype} and {dtype}'
            assert ti.device == device, f'The device of the input tensors should be the same, but got {ti.device} and {device}'
        
        for t in args:
            if dtype is None:
                dtype = t.dtype
            if device is None:
                device = t.device
            
            if isinstance(t, (tuple, list)):
                for ti in t:
                    _asserts(ti, dtype, device)
            else:
                _asserts(t, dtype, device)
                
    def forward(self,
                img_fusion: Tensor,
                boundary_gt: "Tensor | tuple",  # cat([vi, ir]) or tuple(vis, ir)
                fusion_gt: "Tensor"=None,
                mask: Tensor=None) -> tuple[torch.Tensor, dict]:
        if mask is not None:
            self.check_dtype_and_device(img_fusion, boundary_gt, mask)
            with torch.no_grad():
                if self.reduce_label:
                    mask2 = mask
                    mask[mask > 1.] = 1.
                else:
                    mask2 = mask
        else:
            self.check_dtype_and_device(img_fusion, boundary_gt)
        
        wd = self.weight_dict
        loss_intensity = 0
        loss_color = 0
        loss_grad = 0
        loss_fusion = 0
        loss = {}
        
        # split boundary gt
        no_batch_ndim = img_fusion.ndim - 1
        broadcast_fn = lambda x: x.reshape(-1, *[1]*no_batch_ndim)
        if isinstance(boundary_gt, tuple):
            img_B, img_A = boundary_gt
        else:
            img_A, img_B = boundary_gt[:, 3:], boundary_gt[:, :3]
            
        # if has gt (e.g., when we train model on multi-exposure image fusion task)
        if exists(fusion_gt):
            loss_fusion += wd['fusion_gt'] * self.loss_func(img_fusion, fusion_gt)
            
        # ir, vi
        img_A, img_B = self.check_rgb(img_A), self.check_rgb(img_B)
        
        # vgg latent weights
        if self.latent_weighted:
            vi_w, ir_w = self.dynamic_weight(img_A, img_B)
            vi_w = broadcast_fn(vi_w)
            ir_w = broadcast_fn(ir_w)
        else:
            vi_w, ir_w = 1.0, 1.0
        
        # YCbCr decomposition
        # use for intensity, color, and gradient loss
        Y_fusion, Cb_fusion, Cr_fusion = RGB2YCrCb(img_fusion)
        Y_A, Cb_A, Cr_A = RGB2YCrCb(img_A)  # ir
        Y_B, Cb_B, Cr_B = RGB2YCrCb(img_B)  # vi
        Y_joint = torch.max(Y_A, Y_B)
        
        ## intensity and color loss
        if mask is not None:
            # intensity loss
            # 1. || fusion - max(vi, ir) ||                     max fusion
            # 2. || mask * fusion - mask * ir ||                ir loss (pedistrain)
            # 3. || (1 - mask) * fusion - (1 - mask) * vi ||    vi loss (background)
            loss_intensity = wd['inten_f_joint'] * self.loss_func(Y_fusion, Y_joint) + \
                             wd['inten_f_ir'] * ir_w * self.loss_func(mask2 * Y_fusion, mask2 * Y_A) + \
                             wd['inten_f_vi'] * vi_w * self.loss_func(Y_fusion * (1 - mask2), Y_B * (1 - mask2))
            # color loss
            # 1. || (1-mask) * fusion_Cb - (1-mask) * ir_Cb ||          ir_Cb loss
            # 2. || (1-mask) * fusion_Cr - (1-mask) * ir_Cr ||          ir_Cr loss
            if self.color_loss_bg_masked:
                bg_mask = 1 - mask2
            else:
                bg_mask = torch.ones_like(mask2)
            loss_color = wd['color_f_cb'] * vi_w * self.loss_func(Cb_fusion * (1 - mask2), Cb_B * (1 - mask2)) + \
                         wd['color_f_cr'] * ir_w * self.loss_func(Cr_fusion * (1 - mask2), Cr_B * (1 - mask2))
        else:
            loss_intensity = wd['inten_f_joint'] * self.loss_func(Y_fusion, Y_joint) + \
                             wd['inten_f_ir'] * ir_w * self.loss_func(Y_fusion, Y_A) + \
                             wd['inten_f_vi'] * vi_w * self.loss_func(Y_fusion, Y_B)
                             
            # loss_intensity = 10 * ir_w * self.loss_func(Y_fusion, Y_A) + \
            #                  10 * vi_w * self.loss_func(Y_fusion, Y_B)
            loss_color = wd['color_f_cb'] * self.loss_func(Cb_fusion, Cb_B) + \
                         wd['color_f_cr'] * self.loss_func(Cr_fusion, Cr_B)
            
        loss_intensity = loss_intensity.mean()
        loss_color = loss_color.mean()
        
        loss_fusion += loss_intensity + loss_color
        
        ## grad loss
        if self.grad:
            grad_A = self.sobelconv(Y_A)
            grad_B = self.sobelconv(Y_B)
            grad_fusion = self.sobelconv(Y_fusion)

            grad_joint = torch.max(grad_A, grad_B)
            # if mask is not None:
            #     loss_grad += 10 * self.loss_func(grad_fusion, grad_joint) + 40 * self.loss_func(mask * grad_fusion, mask * grad_A)
            # else:
            loss_grad += wd['grad_f_joint'] * self.loss_func(grad_fusion, grad_joint)
            loss_grad = loss_grad.mean()
            
            loss_fusion += loss_grad
            loss.update({'loss_grad': loss_grad})
        
        ## ssim loss
        if self.ssim:
            loss_ssim = wd['ssim_f_joint'] * (self.ssim_func(img_fusion, img_A) + self.ssim_func(img_fusion, img_B))
            loss_fusion += loss_ssim
            loss.update({'loss_ssim' : loss_ssim})
        
        ## tv loss
        if self.tv:
            tv_loss = wd['tv_f'] * self.tv_loss(img_fusion).mean()
            loss_fusion += tv_loss
            loss.update({'tv_loss': tv_loss})
            
        ## correlation loss
        if self.correlation:
            loss_corr = wd['crr_f'] * self.cc_loss(img_A, img_B, img_fusion)
            loss_fusion += loss_corr
            loss.update({'loss_corr': loss_corr})
        
            
        loss.update({'loss_intensity' : loss_intensity,
                     'loss_color' : loss_color,
                     'loss_fusion' : loss_fusion})
            
        return loss_fusion, loss


## EMMA stage two fusion training loss
    
class EMMAFusionLoss(nn.Module):
    def __init__(self, 
                 fusion_model: BaseModel,
                 to_source_A_model: nn.Module,
                 to_source_B_model: nn.Module,
                 A_pretrain_path: str,
                 B_pretrain_path: str,
                 A_model_kwargs: dict,
                 B_model_kwargs: dict,
                 translation_kwargs: dict={},
                 main_once_fusion_loss: Union[callable, nn.Module]=None,
                 refusion_weight: float=0.1,
                 detach_fused: bool=False,
                 translation_weight: float=1.,
                 model_pred_y: bool=True,
                 ):
        super().__init__()
        device = next(fusion_model.parameters()).device
        # device = 'cuda:1'
        
        self.fusion_model = fusion_model
        
        self.A_model = to_source_A_model(**A_model_kwargs).to(device)
        self.A_model.load_state_dict(torch.load(A_pretrain_path))
        self.A_model.eval()
        logger.info(f'load A_model {self.A_model.__class__} done.')
        
        self.B_model = to_source_B_model(**B_model_kwargs).to(device)
        self.B_model.load_state_dict(torch.load(B_pretrain_path))
        self.B_model.eval()
        logger.info(f'load B_model {self.B_model.__class__} done.')
        
        self.shift_n = translation_kwargs.get('shift_num', 3)
        self.rotate_n = translation_kwargs.get('rotate_num', 3)
        self.flip_n = translation_kwargs.get('flip_num', 3)
        logger.info(f'translation params: {translation_kwargs}')
        logger.warning(f'{__class__.__name__}: notice that your batch size will be enlarged by setting shift_n, ' + \
                       f'rotate_n, flip_n, total [red]x{self.shift_n + self.rotate_n + self.flip_n} times[/red]')
        
        # apply some loss to first fusion image
        self.main_once_fusion_loss = main_once_fusion_loss
        
        self.refusion_weight = refusion_weight
        self.detach_fused = detach_fused
        self.translation_weight = translation_weight
        self.model_is_y_pred = model_pred_y
        
    def translation_loss(self, fused_img, s_A, s_B, mask=None):
        # once fused image by outter training loop
        # i.e., fusion_model(s_A, s_B)
        
        # fused image    
        if fused_img.size(1) == 3:  # if input image is rgb
            y_cbcr = kornia.color.rgb_to_ycbcr(fused_img)
            Y, Cb, Cr = torch.split(y_cbcr, 1, dim=1)
            F_to_A = self.A_model(Y).clip(0, 1)
            F_to_B = self.B_model(Y).clip(0, 1)
            
            F_to_A = kornia.color.ycbcr_to_rgb(torch.cat([F_to_A, Cb, Cr], dim=1))
            # F_to_B = kornia.color.ycbcr_to_rgb(torch.cat([F_to_B, Cb, Cr], dim=1))
        else:  # if input image is gray image
            F_to_A = self.A_model(fused_img)
            F_to_B = self.B_model(fused_img)
        
        # translation fused image
        
        # NOTE: note that the implementation should double
        # the computation graph to calculate the refusion loss
        # this may cause the GPU memory issue.
        if self.detach_fused:
            fused_img_detach = fused_img.detach()
        else:
            fused_img_detach = fused_img
        trans_fused_img = self.apply_translation(fused_img_detach)
        
        if fused_img.size(1) == 3:  # if input image is rgb
            y_cbcr = kornia.color.rgb_to_ycbcr(trans_fused_img)
            Y, Cb, Cr = torch.split(y_cbcr, 1, dim=1)
            Ft_to_A = self.A_model(Y).clip(0, 1)
            Ft_to_B = self.B_model(Y).clip(0, 1)
            
            Ft_to_A = kornia.color.ycbcr_to_rgb(torch.cat([Ft_to_A, Cb, Cr], dim=1))
            # Ft_to_B = kornia.color.ycbcr_to_rgb(torch.cat([Ft_to_B, Cb, Cr], dim=1))
        else:  # if input image is gray image
            Ft_to_A = self.A_model(trans_fused_img)
            Ft_to_B = self.B_model(trans_fused_img)
        
        # refusion
        if self.model_is_y_pred:
            Ft_to_A_ycbcr = kornia.color.rgb_to_ycbcr(Ft_to_A)
            _Ft_A_y = Ft_to_A_ycbcr[:, :1]
            _Ft_A_cbcr = Ft_to_A_ycbcr[:, 1:]
        else:
            _Ft_A_y = Ft_to_A
            _Ft_A_cbcr = None
        Ft_refused = self.fusion_model.only_fusion_step(_Ft_A_y, Ft_to_B)
        if self.model_is_y_pred:
            Ft_refused = kornia.color.ycbcr_to_rgb(torch.cat([Ft_refused, _Ft_A_cbcr], dim=1))
        
        # three losse
        # 1. source A loss
        loss_A = self.translation_basic_loss(F_to_A, s_A)
        
        # 2. source B loss
        loss_B = self.translation_basic_loss(F_to_B, s_B)
        
        # 3. refusion loss
        loss_refusion = self.translation_basic_loss(Ft_refused, trans_fused_img) * self.refusion_weight
        
        return loss_A + loss_B + loss_refusion
        
    def translation_basic_loss(self, Ft_to_any, source):
        l1_loss = F.l1_loss(Ft_to_any, source)
        grad_loss = F.l1_loss(spatial_gradient(Ft_to_any),
                              spatial_gradient(source))
        
        return l1_loss + grad_loss
    
    def forward(self,
                fused: torch.Tensor,
                source_AB: "torch.Tensor | tuple[torch.Tensor, torch.Tensor]",
                mask: torch.Tensor=None):
        loss_dict = {}
        
        if isinstance(source_AB, Sequence):
            A, B = source_AB
        elif isinstance(source_AB, torch.Tensor):
            A, B = source_AB[:, :3], source_AB[:, 3:]
        else:
            raise ValueError('source_AB should be a tuple or a tensor')
        
        if self.main_once_fusion_loss:
            fused_loss, _ = self.main_once_fusion_loss(fused, (A, B), mask=mask)
            loss_dict['fusion_loss'] = fused_loss
            
        translation_loss = self.translation_loss(fused, A, B) * self.translation_weight
        loss_dict['tra_loss'] = translation_loss
        
        return fused_loss + translation_loss, loss_dict
    
    def apply_translation(self, x):
        if self.shift_n>0:
            x_shift = self.shift_random(x, self.shift_n)
        if self.rotate_n>0:
            x_rotate = self.rotate_random(x, self.rotate_n)
        if self.flip_n>0:
            x_flip = self.flip_random(x, self.flip_n)

        if self.shift_n>0:
            x = torch.cat((x,x_shift),0)
        if self.rotate_n>0:
            x = torch.cat((x,x_rotate),0)
        if self.flip_n>0:
            x = torch.cat((x,x_flip),0)
            
        return x
    
    @staticmethod  
    def shift_random(x, n_trans=5):
        H, W = x.shape[-2], x.shape[-1]
        assert n_trans <= H - 1 and n_trans <= W - 1, 'n_shifts should less than {}'.format(H-1)
        shifts_row = random.sample(list(np.concatenate([-1*np.arange(1, H), np.arange(1, H)])), n_trans)
        shifts_col = random.sample(list(np.concatenate([-1*np.arange(1, W), np.arange(1, W)])), n_trans)
        x = torch.cat([torch.roll(x, shifts=[sx, sy], dims=[-2,-1]).type_as(x) for sx, sy in zip(shifts_row, shifts_col)], dim=0)
        
        return x

    @staticmethod
    def rotate_random(data, n_trans=5, random_rotate=False):
        if random_rotate:
            theta_list = random.sample(list(np.arange(1, 359)), n_trans)
        else:
            theta_list = np.arange(10, 360, int(360 / n_trans))
        # data = torch.cat([kornia.geometry.rotate(data, torch.Tensor([theta]).type_as(data))for theta in theta_list], dim=0)
        d = []
        for theta in theta_list:
            d.append(kornia.geometry.rotate(data, torch.tensor(theta).to(data)))
        
        return torch.cat(d, dim=0)
    
    @staticmethod
    def flip_random(data, n_trans=3):
        assert n_trans <= 3, 'n_flip should less than 3'
        
        if n_trans>=1:
            data1=kornia.geometry.transform.hflip(data)
        if n_trans>=2:
            data2=kornia.geometry.transform.vflip(data)
            data1=torch.cat((data1,data2),0)
        if n_trans==3:
            data1=torch.cat((data1,kornia.geometry.transform.hflip(data2)),0)        
            
        return data1

    
def get_emma_fusion_loss(fusion_model: nn.Module, 
                         device: str=None,
                         model_is_y_pred: bool=True):
    from utils.utils_modules import TranslationUnet
    
    # color bg may cause object color shift, so we constraint on the whole image
    main_loss = DRMFFusionLoss(reduce_label=False, color_loss_bg_masked=True)
    if device is not None:
        main_loss = main_loss.to(device)
    else:
        main_loss = main_loss.cuda()
        
    return EMMAFusionLoss(
        fusion_model=fusion_model,
        to_source_A_model=TranslationUnet,
        to_source_B_model=TranslationUnet,
        A_pretrain_path='utils/ckpts/Av.pth',
        B_pretrain_path='utils/ckpts/Ai.pth',
        A_model_kwargs={},
        B_model_kwargs={},
        translation_kwargs={'shift_num': 0, 'rotate_num': 1, 'flip_num': 2},
        main_once_fusion_loss=main_loss,
        detach_fused=True,  # avoid GPU OOM
        translation_weight=6.,
        refusion_weight=0.1,
        model_pred_y=model_is_y_pred
    )

# TODO: fusion task: add loss on mask
from typing_extensions import TypedDict, Unpack, TYPE_CHECKING

if TYPE_CHECKING:
    class GetLossKwargsdType(TypedDict):
        latent_weighted: bool
        tv_loss: bool
        grad_loss: bool
        ssim_loss: bool
        tv_loss: bool
        pseudo_l1_const: float
        correlation_loss: bool
        reduce_label: bool
        weight_dict: dict

def get_loss(loss_type, channel=31, **kwargs: "Unpack[GetLossKwargsdType]"):
    
    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "l1":
        criterion = TorchLossWrapper((1.,), l1=nn.L1Loss())
    elif loss_type == "hybrid":
        criterion = HybridL1L2()
    elif loss_type == "smoothl1":
        criterion = nn.SmoothL1Loss()
    elif loss_type == "l1ssim":
        criterion = HybridL1SSIM(channel=channel, weighted_r=(1.0, 0.1))
    elif loss_type == "ssimrmi_fuse":
        criterion = HybridSSIMRMIFuse(weight_ratio=(1, 1), ssim_channel=channel)
    elif loss_type == "pia_fuse":
        # perceptual loss should be less weighted
        criterion = HybridPIALoss(weight_ratio=(3, 7, 20, 10))
    elif loss_type == "charbssim":
        criterion = HybridCharbonnierSSIM(channel=channel, weighted_r=(1.0, 1.0))
    elif loss_type == "ssimsf":
        # YDTR loss
        # not hack weighted ratio
        criterion = HybridSSIMSF(channel=1)
    elif loss_type == "ssimmci":
        criterion = HybridSSIMMCI(channel=1)
    elif loss_type == "mcgmci":
        criterion = HybridMCGMCI(weight_r=(2.0, 1.0))
    elif loss_type == "u2fusion":
        criterion = U2FusionLoss()
    elif loss_type == "cddfusion":
        criterion = CDDFusionLoss(weights=(1.5, 1, 1))
    elif loss_type == "swinfusion":
        criterion = SwinFusionLoss()
    elif loss_type == 'drmffusion':
        criterion = DRMFFusionLoss(latent_weighted=kwargs.pop('latent_weighted', True),
                                   grad_loss=kwargs.pop('grad_loss', True),
                                   ssim_loss=kwargs.pop('ssim_loss', True),
                                   tv_loss=kwargs.pop('tv_loss', False),
                                   pseudo_l1_const=kwargs.pop('pseudo_l1_const', 1.4e-3),
                                   correlation_loss=kwargs.pop('correlation_loss', False),
                                   reduce_label=kwargs.pop('reduce_label', True),
                                   weight_dict=kwargs.pop('weight_dict', None))
    elif loss_type == 'emmafusion':
        criterion = get_emma_fusion_loss(kwargs.pop('fusion_model'),
                                         kwargs.pop('device'),
                                         kwargs.pop('model_is_y_pred'))
    else:
        raise NotImplementedError(f"loss {loss_type} is not implemented")
    return criterion


if __name__ == "__main__":
    # loss = SSIMLoss(channel=31)
    # loss = CharbonnierLoss(eps=1e-3)
    # x = torch.randn(1, 31, 64, 64, requires_grad=True)
    # y = x + torch.randn(1, 31, 64, 64) / 10
    # l = loss(x, y)
    # l.backward()
    # print(l)
    # print(x.grad)

    # import PIL.Image as Image

    # vi = (
    #     np.array(
    #         Image.open(
    #             "/media/office-401/Elements SE/cao/ZiHanCao/datasets/RoadScene_and_TNO/training_data/vi/FLIR_05857.jpg"
    #         ).convert("L")
    #     )
    #     / 255
    # )
    # ir = (
    #     np.array(
    #         Image.open(
    #             "/media/office-401/Elements SE/cao/ZiHanCao/datasets/RoadScene_and_TNO/training_data/ir/FLIR_05857.jpg"
    #         ).convert("L")
    #     )
    #     / 255
    # )

    # torch.cuda.set_device("cuda:0")

    # vi = torch.tensor(vi)[None, None].float()  # .cuda()
    # ir = torch.tensor(ir)[None, None].float()  # .cuda()

    # fuse = ((vi + ir) / 2).repeat_interleave(2, dim=0)
    # fuse.requires_grad_()
    # print(fuse.requires_grad)

    # gt = torch.cat((vi, ir), dim=1).repeat_interleave(2, dim=0)

    # fuse_loss = HybridSSIMRMIFuse(weight_ratio=(1.0, 1.0, 1.0), ssim_channel=1)
    
    torch.cuda.set_device("cuda:1")
    
    class FuseModel:
        def only_fusion_step(self, a, b):
            return a + b
    
    # fuse_loss = DRMFFusionLoss(reduce_label=True).cuda()
    fuse_loss = get_emma_fusion_loss(FuseModel())
    # print(fuse_loss(fused, (vis, ir), mask))
    
    # u2fusion_loss = U2FusionLoss().cuda()
    
    import time
    while True:
        fused = torch.randn(1, 3, 64, 64).cuda().requires_grad_()
        vis = torch.randn(1, 3, 64, 64).cuda().requires_grad_()
        ir = torch.randn(1, 1, 64, 64).cuda().requires_grad_()
        mask = torch.randint(0, 3, (1, 1, 64, 64)).cuda().float()
        
        loss = fuse_loss(fused, (vis, ir))
        loss[0].backward()
        print(loss)
        time.sleep(0.1)
    
    
    # fuse_loss = HybridPIALoss().cuda(1)
    # fuse_loss = CDDFusionLoss()  # .cuda()
    # loss, loss_d = fuse_loss(fuse, gt)
    # loss.backward()
    # print(loss)
    # print(loss_d)

    # print(fuse.grad)

    # mcg_mci_loss = HybridMCGMCI()
    # print(mcg_mci_loss(fuse, gt))
