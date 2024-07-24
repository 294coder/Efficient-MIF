# GPL License
# Copyright (C) UESTC
# All Rights Reserved
#
# @Time    : 2023/4/10 12:37
# @Author  : Xiao Wu and Zihan Cao
# @reference:
#
import sys
from typing import Union
sys.path.append('./')
from utils.misc import dict_to_str
from torchmetrics.functional.image import visual_information_fidelity

from torch import Tensor
    


class AnalysisVISIRAcc(object):
    def __init__(self, unorm=True):
        self.metric_fn = analysis_Reference_fast
        self.unorm_factor = 255 if unorm else 1

        # acc tracker
        self._acc_d = {}
        self._call_n = 0
        self.acc_ave = {}
        self.sumed_acc = {}

    def _average_acc(self, d_ave, n):
        for k in d_ave.keys():
            d_ave[k] /= n
        return d_ave
    
    @property
    def empty_acc(self):
        return {
            'PSNR': 0.0,
            'EN': 0.0,
            'SD': 0.0,
            'SF': 0.0,
            'AG': 0.0,
            'SSIM': 0.0,
            'VIF': 0.0
        }

    def drop_dim(self, x):
        """
        [1, h, w] -> [h, w]
        [c, h, w]: unchange
        """
        assert x.ndim == 3
        
        if x.size(0) == 1:
            return x[0]
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

    def one_batch_call(self, gt: "Tensor | tuple[Tensor]", pred: "Tensor"):
        """call the metric function for one batch

        Args:
            gt (Tensor | tuple[Tensor]): Tensor by catting the vis and ir, or tuple of vis and ir;
            channel for `Tensor` type should be 1+1 or 3+1 (rgb and infared). If tuple, assumed to be (vis, ir).
            pred (Tensor): fused image shaped as [b, 1, h, w] or [b, 3, h, w].
        """
        fusion_chans = pred.size(1)
        if isinstance(gt, Tensor):
            gt_chans = gt.size(1)
            assert gt.shape[-2:]==pred.shape[-2:], f'gt and pred should have same shape,' \
                f'but got gt.shape[-2:]=={gt.shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}'
                
            assert gt_chans == 4 or gt_chans == 2, f'gt.size(1) should be 4 or 2, but got {gt.size(1)}'
            vi, ir = gt.split([fusion_chans, gt_chans - fusion_chans], dim=1)
        else:
            assert len(gt) == 2, f'gt should have 2 element, but got {len(gt)}'
            assert gt[0].shape[-2:] == pred.shape[-2:], f'gt[0] and pred should have same shape,' \
                f'but got gt[0].shape[-2:]=={gt[0].shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}'
            assert gt[1].shape[-2:] == pred.shape[-2:], f'gt[1] and pred should have same shape,' \
                f'but got gt[1].shape[-2:]=={gt[1].shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}'
            vi, ir = gt
        
        b = gt.shape[0]
        gt = gt * self.unorm_factor
        pred = pred * self.unorm_factor

        # input shapes are [B, C, H, W]
        # gt is [b, 2, h, w]
        # pred is [b, 1, h, w]
        batch_acc_d = []
        for vi_i, ir_i, f_i in zip(vi, ir, pred):
            vi_i, ir_i, f_i = map(self.drop_dim, (vi_i, ir_i, f_i))
            acc_d = self.metric_fn(f_i, ir_i, vi_i)
            batch_acc_d.append(acc_d)

        sum_d = self.dict_items_sum(batch_acc_d)
        self._acc_d = sum_d
        self.sumed_acc, self.acc_ave, self._call_n = self.average_all(sum_d, b, self._call_n, self.sumed_acc)

    def __call__(self, gt, pred):
        self.one_batch_call(gt, pred)

    def result_str(self):
        return dict_to_str(self.acc_ave)
    
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





## taken from the CDDFuse repo: https://github.com/Zhaozixiang1228/MMIF-CDDFuse

import numpy as np
import cv2
import sklearn.metrics as skm
from scipy.signal import convolve2d
import math
from skimage.metrics import structural_similarity as ssim

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

class Evaluator():
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None): 
        if imgA is None:
            assert type(imgF) == np.ndarray, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert type(imgF) == type(imgA) == type(imgB) == np.ndarray, 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):  # entropy
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a) / a.shape[0]
        return -sum(h * np.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return np.std(img)

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2))

    @classmethod
    def AG(cls, img):  # Average gradient
        cls.input_check(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)

        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return skm.mutual_info_score(image_F.flatten(), image_A.flatten()) + skm.mutual_info_score(image_F.flatten(),
                                                                                                   image_B.flatten())

    @classmethod
    def MSE(cls, image_F, image_A, image_B):  # MSE
        cls.input_check(image_F, image_A, image_B)
        return (np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        return (rAF + rBF) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return 10 * np.log10(np.max(image_F) ** 2 / cls.MSE(image_F, image_A, image_B))

    @classmethod
    def SCD(cls, image_F, image_A, image_B): # The sum of the correlations of differences
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)))
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
        return corr1 + corr2

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F)+cls.compare_viff(image_B, image_F)

    @classmethod
    def compare_viff(cls,ref, dist): # viff of a pair of pictures
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if np.isnan(vifp):
            return 1.0
        else:
            return vifp

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)

        # 计算QABF
        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return nume / deno

    @classmethod
    def Qabf_getArray(cls,img):
        # Sobel Operator Sobel
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0]=np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls,aA, gA, aF, gF):
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF,AAF,QgAF,QaAF,QAF = np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA)
        GAF[gA>gF]=gF[gA>gF]/gA[gA>gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA <gF] = gA[gA<gF]/gF[gA<gF]
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        QAF = QgAF* QaAF
        return QAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return ssim(image_F,image_A)+ssim(image_F,image_B)


def VIFF(image_F, image_A, image_B):
    refA=image_A
    refB=image_B
    dist=image_F

    sigma_nsq = 2
    eps = 1e-10
    numA = 0.0
    denA = 0.0
    numB = 0.0
    denB = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        # Create a Gaussian kernel as MATLAB's
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        if scale > 1:
            refA = convolve2d(refA, np.rot90(win, 2), mode='valid')
            refB = convolve2d(refB, np.rot90(win, 2), mode='valid')
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
            refA = refA[::2, ::2]
            refB = refB[::2, ::2]
            dist = dist[::2, ::2]

        mu1A = convolve2d(refA, np.rot90(win, 2), mode='valid')
        mu1B = convolve2d(refB, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
        mu1_sq_A = mu1A * mu1A
        mu1_sq_B = mu1B * mu1B
        mu2_sq = mu2 * mu2
        mu1A_mu2 = mu1A * mu2
        mu1B_mu2 = mu1B * mu2
        sigma1A_sq = convolve2d(refA * refA, np.rot90(win, 2), mode='valid') - mu1_sq_A
        sigma1B_sq = convolve2d(refB * refB, np.rot90(win, 2), mode='valid') - mu1_sq_B
        sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
        sigma12_A = convolve2d(refA * dist, np.rot90(win, 2), mode='valid') - mu1A_mu2
        sigma12_B = convolve2d(refB * dist, np.rot90(win, 2), mode='valid') - mu1B_mu2

        sigma1A_sq[sigma1A_sq < 0] = 0
        sigma1B_sq[sigma1B_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        gA = sigma12_A / (sigma1A_sq + eps)
        gB = sigma12_B / (sigma1B_sq + eps)
        sv_sq_A = sigma2_sq - gA * sigma12_A
        sv_sq_B = sigma2_sq - gB * sigma12_B

        gA[sigma1A_sq < eps] = 0
        gB[sigma1B_sq < eps] = 0
        sv_sq_A[sigma1A_sq < eps] = sigma2_sq[sigma1A_sq < eps]
        sv_sq_B[sigma1B_sq < eps] = sigma2_sq[sigma1B_sq < eps]
        sigma1A_sq[sigma1A_sq < eps] = 0
        sigma1B_sq[sigma1B_sq < eps] = 0

        gA[sigma2_sq < eps] = 0
        gB[sigma2_sq < eps] = 0
        sv_sq_A[sigma2_sq < eps] = 0
        sv_sq_B[sigma2_sq < eps] = 0

        sv_sq_A[gA < 0] = sigma2_sq[gA < 0]
        sv_sq_B[gB < 0] = sigma2_sq[gB < 0]
        gA[gA < 0] = 0
        gB[gB < 0] = 0
        sv_sq_A[sv_sq_A <= eps] = eps
        sv_sq_B[sv_sq_B <= eps] = eps

        numA += np.sum(np.log10(1 + gA * gA * sigma1A_sq / (sv_sq_A + sigma_nsq)))
        numB += np.sum(np.log10(1 + gB * gB * sigma1B_sq / (sv_sq_B + sigma_nsq)))
        denA += np.sum(np.log10(1 + sigma1A_sq / sigma_nsq))
        denB += np.sum(np.log10(1 + sigma1B_sq / sigma_nsq))

    vifpA = numA / denA
    vifpB =numB / denB

    if np.isnan(vifpA):
        vifpA=1
    if np.isnan(vifpB):
        vifpB = 1
    return vifpA+vifpB



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
    # from scipy import io as sio
    #
    # data = sio.loadmat('./test_IRF.mat')
    # im1 = data['image1']
    # im2 = data['image2']
    # image_f = data['image_f']
    # print(im1.shape, im2.shape, image_f.shape, im1.max(), im2.max(), image_f.max())
    # im1 = torch.from_numpy(im1).float() * 255
    # im2 = torch.from_numpy(im2).float() * 255
    # image_f = torch.from_numpy(image_f).float()
    #
    # analysis_Reference_fast(image_f, im1, im2)

    # f = torch.randint(0, 255, (256, 256), dtype=torch.float)
    # vi = torch.randint(0, 255, (256, 256), dtype=torch.float)
    # ir_RS = torch.randint(0, 255, (256, 256), dtype=torch.float)

    # analysis_Reference_fast(f, vi, ir_RS)

    # gt = (torch.randn(2, 2, 256, 256) + 1) / 2
    # sr = (torch.randn(2, 1, 256, 256) + 1) / 2
    # analyser = AnalysisVISIRAcc()

    # # analyser(gt, sr)
    # # print(analyser.result_str())

    # analyser(gt, sr)
    # print(analyser.result_str())
    
    
    # gt = (torch.randn(2, 2, 256, 256) + 1) / 2
    # sr = (torch.randn(2, 1, 256, 256) + 1) / 2
    # analyser2 = AnalysisVISIRAcc()
    
    # analyser2(gt, sr)
    # print(analyser2.result_str())
    
    # gt = (torch.randn(2, 2, 256, 256) + 1) / 2
    # sr = (torch.randn(2, 1, 256, 256) + 1) / 2
    # analyser3 = AnalysisVISIRAcc()
    
    # analyser3(gt, sr)
    # print(analyser3.result_str())
    
    # analyser.ave_result_with_other_analysors([analyser2, analyser3], ave_to_self=True)
    # print(analyser.result_str())
    
    
    fused = np.random.randn(256, 256)
    vi = np.random.randn(256, 256)
    ir = np.random.randn(256, 256)
    
    print(f'SCD: {Evaluator.SCD(fused, vi, ir)}')
    
    
