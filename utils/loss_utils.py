from functools import partial
import random
from typing import Sequence, Union, TYPE_CHECKING
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

import sys
sys.path.append('./')

from utils.misc import is_main_process, exists, default

from utils.log_utils import easy_logger

if TYPE_CHECKING:
    from model.base_model import BaseModel

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
def ave_multi_rank_dict(rank_loss_dict: "list[dict] | dict"):
    # type is dict is only one process
    assert isinstance(rank_loss_dict, (list, dict)), 'rank_loss_dict must be a list or a dict'
    
    if isinstance(rank_loss_dict, dict):
        return rank_loss_dict
    
    n = len(rank_loss_dict)
    if n == 1:
        return rank_loss_dict[0]
        
    ave_dict = {}
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
    
    fuse_loss = DRMFFusionLoss(grad_op='sobel', reduce_label=True, pseudo_l1_const=0.).cuda()
    # fuse_loss = get_emma_fusion_loss(FuseModel())
    # print(fuse_loss(fused, (vis, ir), mask))
    
    # u2fusion_loss = U2FusionLoss().cuda()
    
    import time
    while True:
        fused = torch.randn(1, 3, 64, 64).cuda().requires_grad_()
        vis = torch.randn(1, 3, 64, 64).cuda().requires_grad_()
        ir = torch.randn(1, 3, 64, 64).cuda().requires_grad_()
        mask = torch.randint(0, 3, (1, 1, 64, 64)).cuda().float()
        
        loss = fuse_loss(fused, (vis, ir), mask=mask)
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
