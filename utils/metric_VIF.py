# GPL License
# Copyright (C) UESTC
# All Rights Reserved
#
# @Time    : 2023/4/10 12:37
# @Author  : Xiao Wu and Zihan Cao
# @reference:
#
import sys
from collections.abc import Sequence
import torch
from torch import Tensor
import numpy as np
from functools import partial
from tqdm import tqdm
from torchmetrics.functional.image import visual_information_fidelity
sys.path.append('./')
    
from utils.misc import dict_to_str
from utils.log_utils import easy_logger
from utils._metric_VIS_IR import evaluate_fast_metric_numpy, evaluate_fast_metric_torch

type predType = Tensor | np.ndarray
type gtType = (Tensor | tuple[Tensor] | dict[str, Tensor] | np.ndarray |
              Sequence[Tensor, Tensor] | Sequence[np.ndarray, np.ndarray] |
              dict[str, np.ndarray])

logger = easy_logger(func_name='AnalysisVISIRAcc')

_NEW_METRICS = ['EN', 'SD', 'SF', 'AG', 'MI', 'MSE', 'CC', 'PSNR', 'SCD', 'VIF', 'Qabf', 'SSIM']
_OLD_METRICS = ['PSNR', 'EN', 'SD', 'SF', 'AG', 'SSIM', 'VIF']

class AnalysisVISIRAcc(object):
    def __init__(self, 
                 unorm: bool=True,
                 legacy_metric: bool=False,
                 progress_bar: bool=False,
                 results_decimals: int=4,
                 test_metrics: str | list[str] = "all",
                 implem_by: str='torch',
                 ):
        self.unorm_factor = 255 if unorm else 1
        if self.unorm_factor != 255:
            logger.warning('image range should be [0, 255] for VIF metric, ' + \
                           'but got unorm_factor={self.unorm_factor}.')
        self.legacy_metric = legacy_metric
        self._tested_metrics = test_metrics
        self.implem_by = implem_by
        self._check_test_metrics(test_metrics)
        
        if self.legacy_metric:
            logger.warning('using legacy metric which is not recommended and it is implemented by numpy which is slow')
        if self.legacy_metric:
            self.metric_fn = analysis_Reference_fast
        else:
            if implem_by == 'torch':
                self.metric_fn = evaluate_fast_metric_torch
            else:
                logger.warning(f'using numpy implementation for metric which is [i]slow[/i].')
                logger.warning('numpy implementaion may differ from torch implementaion,',
                               'we recommend using torch implementaion.')
                self.metric_fn = evaluate_fast_metric_numpy
        self.progress_bar = progress_bar
        self.results_decimals = results_decimals
        
        # acc tracker
        self._acc_d = {}
        self._call_n = 0
        self.acc_ave = {}
        self.sumed_acc = {}
    
    def _check_test_metrics(self, test_metrics):
        if test_metrics != 'all':
            if self.legacy_metric:
                for m in test_metrics:
                    assert m in _OLD_METRICS, f'metric {m} is not in {_OLD_METRICS}'
            else:
                for m in test_metrics:
                    assert m in _NEW_METRICS, f'metric {m} is not in {_NEW_METRICS}'

    @property
    def tested_metrics(self):
        if self._tested_metrics == 'all':
            if self.legacy_metric:
                return _OLD_METRICS
            else:
                return _NEW_METRICS
            
        return self._tested_metrics

    def _average_acc(self, d_ave, n):
        for k in d_ave.keys():
            d_ave[k] /= n
        return d_ave
    
    @property
    def empty_acc(self):
        return {k: 0. for k in self.tested_metrics}

    def drop_dim(self, x: "torch.Tensor", to_numpy=False):
        """
        [1, h, w] -> [h, w]
        [c, h, w]: unchange
        """
        assert x.ndim in (2, 3), f'must be 2 or 3 number of dimensions, but got {x.ndim}'
        
        if x.size(0) == 1:
            if to_numpy:
                return x[0].detach().cpu().numpy()
            else:
                return x[0]
        else:
            if to_numpy:
                return x.detach().cpu().numpy()
            else:
                return x
            
    @staticmethod
    def dict_items_sum(b_acc_d: list):
        sum_d = {}
        for acc_d in b_acc_d:
            for k, v in acc_d.items():
                sum_d[k] = sum_d.get(k, 0) + v
        return sum_d

    @staticmethod
    def average_all(sum_d: dict, bs: int, prev_call_n: int, prev_sumed_acc: dict):
        call_n = prev_call_n + bs
        summed_acc = {}
        acc_ave = {}
        
        for k, v in sum_d.items():
            summed_acc[k] = prev_sumed_acc.get(k, 0) + v
            acc_ave[k] = summed_acc[k] / call_n
            
        return summed_acc, acc_ave, call_n
    
    def type_check(self, x: Tensor | tuple[Tensor]):
        def _inner_type_check(x):
            if torch.is_tensor(x):
                if x.dtype == torch.uint8:
                    assert self.unorm_factor == 1, f'unorm_factor should be 1 for uint8 input, but got {self.unorm_factor}'
                    return x.float()
                else:
                    return x
            
        if isinstance(x, (tuple, list)):
            typed_x = []
            for xi in x:
                typed_x.append(_inner_type_check(xi))
            return typed_x
        else:
            return _inner_type_check(x)

    def one_batch_call(self,
                       gt: gtType,
                       pred: predType):
        """call the metric function for one batch

        Args:
            gt (Tensor | tuple[Tensor]): Tensor by catting the vis and ir, or tuple of vis and ir;
            channel for `Tensor` type should be 1+1 or 3+1 (rgb and infared). If tuple, assumed to be (vis, ir).
            pred (Tensor): fused image shaped as [b, 1, h, w] or [b, 3, h, w].
        """
        fusion_chans = pred.size(1)
        is_rgb = fusion_chans == 3
        
        if isinstance(gt, Tensor):
            gt_chans = gt.size(1)
            assert gt.shape[-2:]==pred.shape[-2:], f'gt and pred should have same shape,' \
                f'but got gt.shape[-2:]=={gt.shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}'
                
            assert gt_chans == 4 or gt_chans == 2, f'gt.size(1) should be 4 or 2, but got {gt.size(1)}'
            vi, ir = gt.split([fusion_chans, gt_chans - fusion_chans], dim=1)
        elif isinstance(gt, (tuple, list)):
            assert len(gt) == 2, f'gt should have 2 element, but got {len(gt)}'
            assert gt[0].shape[-2:] == pred.shape[-2:], f'gt[0] and pred should have same shape,' \
                f'but got gt[0].shape[-2:]=={gt[0].shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}'
            assert gt[1].shape[-2:] == pred.shape[-2:], f'gt[1] and pred should have same shape,' \
                f'but got gt[1].shape[-2:]=={gt[1].shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}'
            vi, ir = gt
        elif isinstance(gt, dict):
            vi = gt['vi']
            ir = gt['ir']
            assert vi.shape[-2:] == pred.shape[-2:], f'vi and pred should have same shape,' \
                f'but got vi.shape[-2:]=={vi.shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}'
            assert ir.shape[-2:] == pred.shape[-2:], f'ir and pred should have same shape,' \
                f'but got ir.shape[-2:]=={ir.shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}'
        else:
            raise ValueError(f'gt should be Tensor or tuple of Tensor, but got {type(gt)}')
        
        b = vi.shape[0]
        vi, ir, pred = map(self.type_check, (vi, ir, pred))
        vi = (vi * self.unorm_factor).clip(0, 255)
        ir = (ir * self.unorm_factor).clip(0, 255)
        pred = (pred * self.unorm_factor).clip(0, 255)

        # input shapes are [B, C, H, W]
        # gt is [b, 2, h, w]
        # pred is [b, 1, h, w]
        batch_acc_d = []
        tbar = tqdm(zip(vi, ir, pred), total=b, disable=not self.progress_bar, leave=False)
        for vi_i, ir_i, f_i in tbar:
            if self.legacy_metric:
                self.metric_fn: analysis_Reference_fast
                # TODO: need test
                assert not is_rgb, 'legacy metric only support gray input'
                vi_i, ir_i, f_i = map(self.drop_dim, (vi_i, ir_i, f_i))
                acc_d = self.metric_fn(f_i, ir_i, vi_i)
            else:
                self.metric_fn: evaluate_fast_metric_numpy
                if is_rgb:
                    _acc_ds = []
                    for j in range(3):
                        _to_numpy = True if self.implem_by == 'numpy' else False
                        fused, vis, ir = map(partial(self.drop_dim, to_numpy=_to_numpy), (f_i[j], vi_i[j], ir_i[0]))
                        _acc_ds.append(self.metric_fn(fused, vis, ir))
                    acc_d = self._mean_dict(_acc_ds)
                else:
                    fused, vis, ir = map(partial(self.drop_dim, to_numpy=True), (f_i[0], vi_i[0], ir_i[0]))
                    acc_d = self.metric_fn(fused, vis, ir)
                        
            batch_acc_d.append(acc_d)

        sum_d = self.dict_items_sum(batch_acc_d)
        self._acc_d = sum_d
        self.sumed_acc, self.acc_ave, self._call_n = self.average_all(sum_d, b, self._call_n, self.sumed_acc)

    @staticmethod
    def _mean_dict(d: "list[dict]"):
        mean_d = {}
        keys = d[0].keys()
        for k in keys:
            mean_d[k] = sum([d_i[k] for d_i in d]) / len(d)
            
        return mean_d

    def __call__(self, gt: gtType, pred: predType):
        self.one_batch_call(gt, pred)

    def result_str(self):
        return dict_to_str(self.acc_ave, decimals=self.results_decimals)
    
    def __repr__(self):
        return (f'AnalysisVISIRAcc(unorm_factor={self.unorm_factor}, legacy_metric={self.legacy_metric}) \n' +
                f'Current result: {self.result_str()}')
            
    @property
    def last_acc(self):
        return self._acc_d
    
    def ave_result_with_other_analysors(self, 
                                       analysors: "Union[list[AnalysisVISIRAcc], AnalysisVISIRAcc]",
                                       ave_to_self: bool=False):
        if isinstance(analysors, AnalysisVISIRAcc):
            analysors = [analysors]
        
        for analysor in analysors:
            assert isinstance(analysor, AnalysisVISIRAcc), f'analysors should be list of `AnalysisVISIRAcc`, but got {type(analysor)}'
            assert analysor._call_n > 0, 'analysor should be called at least once'
            assert list(analysor.acc_ave.keys()) == list(self.acc_ave.keys()), f'analysor should have same keys, but one is {list(analysor.acc_ave.keys())}, \
                                                                                 the other is {list(self.acc_ave.keys())}'

        sum_d = self.dict_items_sum([a.sumed_acc for a in analysors])
        call_n_times = sum([a._call_n for a in analysors])
        sumed_acc, acc_ave, call_n = self.average_all(sum_d, call_n_times, self._call_n, self.sumed_acc)
        
        if ave_to_self:
            self.sumed_acc = sumed_acc
            self.acc_ave = acc_ave
            self._call_n = call_n
        
        return acc_ave




########## ================================== legacy code ================================== ##########

# source code from xiao-woo

#! old previous callable functions

#########
# metric helpers
#########

import torch
from torch.nn import functional as F


def cal_PSNR(A, B, F):
    [m, n] = F.shape
    MSE_AF = torch.sum((F - A) ** 2) / (m * n)
    MSE_BF = torch.sum((F - B) ** 2) / (m * n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    PSNR = 20 * torch.log10(255 / torch.sqrt(MSE))

    return PSNR


def cal_SD(F):
    [m, n] = F.shape
    u = torch.mean(F)

    # 原版 is wrong
    # tmp = (F - u).numpy()
    # tmp2 = np.round(tmp.clip(0)).astype(np.uint16)
    # SD = np.sqrt(np.sum((tmp2 ** 2).clip(0, 255)) / (m * n))

    SD = torch.sqrt(torch.sum((F - u) ** 2) / (m * n))

    return SD

def cal_VIF(A: Tensor, B: Tensor, F: Tensor):
    def check_dims(x):
        ndim = x.ndim
        if ndim == 3:  # [c, h, w]
            x = x[None]
        elif ndim == 2:
            x = x[None, None]
        elif ndim == 4:
            pass
        else:
            raise ValueError(f'x.ndim should be 2 or 3, but got {ndim}')
        
        return x
    
    A, B, F = map(check_dims, (A, B, F))
        
    fusion_chans = F.size(1)
    if A.size(1) != fusion_chans:
        assert A.size(1) == 1
        A = A.expand_as(F)
    if B.size(1) != fusion_chans:
        assert B.size(1) == 1
        B = B.expand_as(F)
        
    # x: [b, c, h, w]
    assert A.size(1) == B.size(1) == F.size(1), f'A.size(1) should be equal to B.size(1) and F.size(1), ' \
                f'but got A.size(1)={A.size(1)}, B.size(1)={B.size(1)}, F.size(1)={F.size(1)}'
            
    vif_fn = visual_information_fidelity
    return vif_fn(F, A) + vif_fn(F, B)

def cal_EN(I):
    p = torch.histc(I, 256)
    p = p[p != 0]
    p = p / torch.numel(I)
    E = -torch.sum(p * torch.log2(p))
    return E


def cal_SF(MF):
    # RF = MF[]#diff(MF, 1, 1);
    [m, n] = MF.shape
    RF = MF[:m - 1, :] - MF[1:]
    RF1 = torch.sqrt(torch.mean(RF ** 2))
    CF = MF[:, :n - 1] - MF[:, 1:]  # diff(MF, 1, 2)
    CF1 = torch.sqrt(torch.mean(CF ** 2))
    SF = torch.sqrt(RF1 ** 2 + CF1 ** 2)

    return SF


def analysis_Reference_fast(image_f: "Tensor",
                            image_ir: "Tensor",
                            image_vis: "Tensor"):
    # shapes are [c, h, w], channel is 1 or 3
    # image_f: 0-255
    # image_ir: 0-255
    # image_vis: 0-255

    if image_f.ndim == 2:
        PSNR = cal_PSNR(image_ir, image_vis, image_f)
        SD = cal_SD(image_f)
        EN = cal_EN(image_f)
        SF = cal_SF(image_f / 255.0)
        AG = cal_AG(image_f)
        SSIM = cal_SSIM(image_ir, image_vis, image_f)
    elif image_f.ndim == 3:  # [c, h, w]
        # compute per channel
        fusion_chans = image_f.size(0)
        assert image_ir.ndim == 2, 'force to expand ir to fusion channel'
        image_ir = image_ir[None].expand_as(image_f)  # [c, h, w]
        PSNRs, SDs, ENs, SFs, AGs, SSIMs = [], [], [], [], [], []
        for i in range(fusion_chans):
            PSNR = cal_PSNR(image_ir[i], image_vis[i], image_f[i])
            SD = cal_SD(image_f[i])
            EN = cal_EN(image_f[i])
            SF = cal_SF(image_f[i] / 255.0)
            AG = cal_AG(image_f[i])
            SSIM = cal_SSIM(image_ir[i], image_vis[i], image_f[i])
            
            PSNRs.append(PSNR)
            SDs.append(SD)
            ENs.append(EN)
            SFs.append(SF)
            AGs.append(AG)
            SSIMs.append(SSIM)
        PSNR, SD, EN, SF, AG, SSIM = [sum(x) / len(x) for x in [PSNRs, SDs, ENs, SFs, AGs, SSIMs]]
    
    # taken batched tensors (batching inside the function)
    VIF = cal_VIF(image_ir, image_vis, image_f)

    return dict(
        PSNR=PSNR.item(),
        EN=EN.item(),
        SD=SD.item(),
        SF=SF.item(),
        AG=AG.item(),
        SSIM=SSIM.item(),
        VIF=VIF.item()
    )


def cal_AG(img):
    if len(img.shape) == 2:
        [r, c] = img.shape
        [dzdx, dzdy] = torch.gradient(img)
        s = torch.sqrt((dzdx ** 2 + dzdy ** 2) / 2)
        g = torch.sum(s) / ((r - 1) * (c - 1))

    else:
        [r, c, b] = img.shape
        g = torch.zeros(b)
        for k in range(b):
            band = img[:, :, k]
            [dzdx, dzdy] = torch.gradient(band)
            s = torch.sqrt((dzdx ** 2 + dzdy ** 2) / 2)
            g[k] = torch.sum(s) / ((r - 1) * (c - 1))
    return torch.mean(g)


def _ssim(img1, img2):
    device = img1.device
    img1 = img1.float()
    img2 = img2.float()

    channel = img1.shape[1]
    max_val = 1
    _, c, w, h = img1.size()
    window_size = min(w, h, 11)
    sigma = 1.5 * window_size / 11

    # 不加这个,对应matlab的quality_assess的ssim指标
    # pad_size = [window_size//2]*4
    # img1 = F.pad(img1, mode='replicate', pad=pad_size)
    # img2 = F.pad(img2, mode='replicate', pad=pad_size)

    window = create_window(window_size, sigma, channel).to(device)
    mu1 = F.conv2d(img1, window, groups=channel)  # , padding=window_size // 2
    mu2 = F.conv2d(img2, window, groups=channel)  # , padding=window_size // 2

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=channel) - mu1_sq  # , padding=window_size // 2
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channel) - mu2_sq  # , padding=window_size // 2
    sigma12 = F.conv2d(img1 * img2, window, groups=channel) - mu1_mu2  # , padding=window_size // 2
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    V1 = 2.0 * sigma12 + C2
    V2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
    t = ssim_map.shape
    return ssim_map.mean(2).mean(2)


import math
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def cal_SSIM(im1, im2, image_f):
    # h, w -> 2, h, w -> b, 2, h, w
    img_Seq = torch.stack([im1, im2])
    image_f = image_f.unsqueeze(0).repeat([img_Seq.shape[0], 1, 1])
    return torch.mean(_ssim(img_Seq.unsqueeze(0) / 255.0, image_f / 255.0))


if __name__ == '__main__':
    # read
    from torchvision.io import read_image
    
    # check shape
    # fused = torch.rand(2, 3, 256, 256)
    # gt = torch.rand(2, 4, 256, 256)
    fused = read_image('/Data3/cao/ZiHanCao/exps/panformer/visualized_img/panRWKV_v8_cond_norm/msrs_v2/00004N.png')[None].float() / 255.
    ir = read_image("/Data3/cao/ZiHanCao/datasets/MSRS/test/ir/00004N.jpg")[None].float() / 255.
    vi = read_image("/Data3/cao/ZiHanCao/datasets/MSRS/test/vi/00004N.jpg")[None].float() / 255.
    
    analyser = AnalysisVISIRAcc(test_metrics='all')
    
    import time
    from tqdm import trange

    for _ in range(2):
        analyser((vi, ir), fused)
        
    t1 = time.time()
    for _ in trange(10):
        analyser((vi, ir), fused)
    
    # print(analyser)
    print(time.time() - t1)