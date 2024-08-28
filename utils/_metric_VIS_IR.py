import torch
import numpy as np
import cv2
import sklearn.metrics as skm
from scipy.signal import convolve2d
import math
import numba
from skimage.metrics import structural_similarity as ssim


def EN(img):  # entropy
    a = np.uint8(np.round(img)).flatten()
    h = np.bincount(a) / a.shape[0]
    return -sum(h * np.log2(h + (h == 0)))

def EN_torch(img: torch.Tensor):
    a = img.flatten().round().long()
    h = torch.bincount(a) / a.shape[0]
    return -torch.sum(h * torch.log2(h + (h == 0)))


def SD(img):
    return np.std(img)

def SD_torch(img):
    return torch.std(img)

def SF(img):
    return np.sqrt(
        np.mean((img[:, 1:] - img[:, :-1]) ** 2)
        + np.mean((img[1:, :] - img[:-1, :]) ** 2)
    )
    
def SF_torch(img):
    return torch.sqrt(
        torch.mean((img[:, 1:] - img[:, :-1]) ** 2)
        + torch.mean((img[1:, :] - img[:-1, :]) ** 2)
    )

def AG(img):  # Average gradient
    Gx, Gy = np.zeros_like(img), np.zeros_like(img)

    Gx[:, 0] = img[:, 1] - img[:, 0]
    Gx[:, -1] = img[:, -1] - img[:, -2]
    Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

    Gy[0, :] = img[1, :] - img[0, :]
    Gy[-1, :] = img[-1, :] - img[-2, :]
    Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
    return np.mean(np.sqrt((Gx**2 + Gy**2) / 2))

def AG_torch(img):
    Gx, Gy = torch.zeros_like(img), torch.zeros_like(img)

    Gx[:, 0] = img[:, 1] - img[:, 0]
    Gx[:, -1] = img[:, -1] - img[:, -2]
    Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

    Gy[0, :] = img[1, :] - img[0, :]
    Gy[-1, :] = img[-1, :] - img[-2, :]
    Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
    return torch.mean(torch.sqrt((Gx**2 + Gy**2) / 2))

def MI(image_F: np.ndarray, image_A, image_B):
    image_F = image_F.astype(np.int32)
    image_A = image_A.astype(np.int32)
    image_B = image_B.astype(np.int32)
    return skm.mutual_info_score(
        image_F.flatten(), image_A.flatten()
    ) + skm.mutual_info_score(image_F.flatten(), image_B.flatten())
    
from torchmetrics.functional.clustering.mutual_info_score import mutual_info_score

def MI_torch(image_F, image_A, image_B):
    image_F = image_F.int()
    image_A = image_A.int()
    image_B = image_B.int()
    return mutual_info_score(image_F.flatten(), image_A.flatten()) + \
           mutual_info_score(image_F.flatten(), image_B.flatten())

def MSE(image_F, image_A, image_B):  # MSE
    return (np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2

def MSE_torch(image_F, image_A, image_B):
    return (torch.mean((image_A - image_F) ** 2) + torch.mean((image_B - image_F) ** 2)) / 2

def CC(image_F, image_A, image_B):
    rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
        (np.sum((image_A - np.mean(image_A)) ** 2))
        * (np.sum((image_F - np.mean(image_F)) ** 2))
    )
    rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
        (np.sum((image_B - np.mean(image_B)) ** 2))
        * (np.sum((image_F - np.mean(image_F)) ** 2))
    )
    return (rAF + rBF) / 2

def CC_torch(image_F, image_A, image_B):
    rAF = torch.sum((image_A - torch.mean(image_A)) * (image_F - torch.mean(image_F))) / torch.sqrt(
        (torch.sum((image_A - torch.mean(image_A)) ** 2))
        * (torch.sum((image_F - torch.mean(image_F)) ** 2))
    )
    rBF = torch.sum((image_B - torch.mean(image_B)) * (image_F - torch.mean(image_F))) / torch.sqrt(
        (torch.sum((image_B - torch.mean(image_B)) ** 2))
        * (torch.sum((image_F - torch.mean(image_F)) ** 2))
    )
    return (rAF + rBF) / 2


def PSNR(image_F, image_A, image_B):
    return 10 * np.log10(np.max(image_F) ** 2 / MSE(image_F, image_A, image_B))

def PSNR_torch(image_F, image_A, image_B):
    return 10 * torch.log10(torch.max(image_F) ** 2 / MSE_torch(image_F, image_A, image_B))

def SCD(image_F, image_A, image_B):  # The sum of the correlations of differences
    imgF_A = image_F - image_A
    imgF_B = image_F - image_B
    corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
        (np.sum((image_A - np.mean(image_A)) ** 2))
        * (np.sum((imgF_B - np.mean(imgF_B)) ** 2))
    )
    corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
        (np.sum((image_B - np.mean(image_B)) ** 2))
        * (np.sum((imgF_A - np.mean(imgF_A)) ** 2))
    )
    return corr1 + corr2

def SCD_torch(image_F, image_A, image_B):
    imgF_A = image_F - image_A
    imgF_B = image_F - image_B
    corr1 = torch.sum((image_A - torch.mean(image_A)) * (imgF_B - torch.mean(imgF_B))) / torch.sqrt(
        (torch.sum((image_A - torch.mean(image_A)) ** 2))
        * (torch.sum((imgF_B - torch.mean(imgF_B)) ** 2))
    )
    corr2 = torch.sum((image_B - torch.mean(image_B)) * (imgF_A - torch.mean(imgF_A))) / torch.sqrt(
        (torch.sum((image_B - torch.mean(image_B)) ** 2))
        * (torch.sum((imgF_A - torch.mean(imgF_A)) ** 2))
    )
    return corr1 + corr2


def VIFF(image_F, image_A, image_B):
    return compare_viff(image_A, image_F) + compare_viff(image_B, image_F)

from torchmetrics.functional.image.vif import visual_information_fidelity

def VIFF_torch(image_F, image_A, image_B):
    image_F = image_F[None, None]
    image_A = image_A[None, None]
    image_B = image_B[None, None]
    return visual_information_fidelity(image_A, image_F) + visual_information_fidelity(image_B, image_F)

def compare_viff(ref, dist):  # viff of a pair of pictures
    sigma_nsq = 2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        # Create a Gaussian kernel as MATLAB's
        m, n = [(ss - 1.0) / 2.0 for ss in (N, N)]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2.0 * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        win2 = np.rot90(win, 2)
        
        if scale > 1:
            ref = convolve2d(ref, win2, mode="valid")
            dist = convolve2d(dist, win2, mode="valid")
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, win2, mode="valid")
        mu2 = convolve2d(dist, win2, mode="valid")
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = convolve2d(ref * ref, win2, mode="valid") - mu1_sq
        sigma2_sq = convolve2d(dist * dist, win2, mode="valid") - mu2_sq
        sigma12 = convolve2d(ref * dist, win2, mode="valid") - mu1_mu2

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


def Qabf(image_F, image_A, image_B):
    gA, aA = Qabf_getArray(image_A)
    gB, aB = Qabf_getArray(image_B)
    gF, aF = Qabf_getArray(image_F)
    QAF = Qabf_getQabf(aA, gA, aF, gF)
    QBF = Qabf_getQabf(aB, gB, aF, gF)

    # 计算QABF
    deno = np.sum(gA + gB)
    nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
    return nume / deno

def Qabf_torch(image_F, image_A, image_B):
    gA, aA = Qabf_getArray_torch(image_A)
    gB, aB = Qabf_getArray_torch(image_B)
    gF, aF = Qabf_getArray_torch(image_F)
    QAF = Qabf_getQabf_torch(aA, gA, aF, gF)
    QBF = Qabf_getQabf_torch(aB, gB, aF, gF)

    # 计算QABF
    deno = torch.sum(gA + gB)
    nume = torch.sum(torch.mul(QAF, gA) + torch.mul(QBF, gB))
    return nume / deno

def Qabf_getArray(img: np.ndarray):
    # Sobel Operator Sobel
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    SAx: np.ndarray = convolve2d(img, h3, mode="same")
    SAy: np.ndarray = convolve2d(img, h1, mode="same")
    gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
    aA = np.zeros_like(img)
    aA[SAx == 0] = math.pi / 2
    aA[SAx != 0] = np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
    return gA, aA

def Qabf_getArray_torch(img: torch.Tensor):
    device = img.device
    img = img[None, None]
    
    # Sobel Operator Sobel
    h1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).to(device)
    h2 = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32).to(device)
    h3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(device)

    SAx: torch.Tensor = torch.nn.functional.conv2d(img, h3.unsqueeze(0).unsqueeze(0), padding=1)[0, 0]
    SAy: torch.Tensor = torch.nn.functional.conv2d(img, h1.unsqueeze(0).unsqueeze(0), padding=1)[0, 0]
    gA = torch.sqrt(torch.mul(SAx, SAx) + torch.mul(SAy, SAy))
    aA = torch.zeros_like(img[0, 0])
    aA[SAx == 0] = math.pi / 2
    aA[SAx != 0] = torch.atan(SAy[SAx != 0] / SAx[SAx != 0])
    return gA, aA

def Qabf_getQabf(aA, gA, aF, gF):
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8
    GAF, AAF, QgAF, QaAF, QAF = (
        np.zeros_like(aA),
        np.zeros_like(aA),
        np.zeros_like(aA),
        np.zeros_like(aA),
        np.zeros_like(aA),
    )
    GAF[gA > gF] = gF[gA > gF] / gA[gA > gF]
    GAF[gA == gF] = gF[gA == gF]
    GAF[gA < gF] = gA[gA < gF] / gF[gA < gF]
    AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
    QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
    QAF = QgAF * QaAF
    return QAF

def Qabf_getQabf_torch(aA, gA, aF, gF):
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8
    GAF, AAF, QgAF, QaAF, QAF = (
        torch.zeros_like(aA),
        torch.zeros_like(aA),
        torch.zeros_like(aA),
        torch.zeros_like(aA),
        torch.zeros_like(aA),
    )
    GAF[gA > gF] = gF[gA > gF] / gA[gA > gF]
    GAF[gA == gF] = gF[gA == gF]
    GAF[gA < gF] = gA[gA < gF] / gF[gA < gF]
    AAF = 1 - torch.abs(aA - aF) / (math.pi / 2)
    QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))
    QAF = QgAF * QaAF
    return QAF


def SSIM(image_F, image_A, image_B):
    # input_check(image_F, image_A, image_B)
    return ssim(image_F, image_A, data_range=255) + ssim(
        image_F, image_B, data_range=255
    )
    
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_torch

def SSIM_torch(image_F, image_A, image_B):
    image_F = image_F[None, None]
    image_A = image_A[None, None]
    image_B = image_B[None, None]
    
    return ssim_torch(image_F, image_A, data_range=255) + ssim_torch(
        image_F, image_B, data_range=255
    )


def evaluate_fast_metric_numpy(
    image_f, image_ir, image_vis, metrics: "list[str] | str" = "all"
) -> dict:
    # shapes are [c, h, w], channel is 1 or 3
    # image_f: 0-255

    if "all" == metrics:
        metrics = [
            "EN",
            "SD",
            "SF",
            "AG",
            "MI",
            "MSE",
            "CC",
            "PSNR",
            "SCD",
            "VIF",
            "Qabf",
            "SSIM",
        ]

    results = {}
    if "EN" in metrics:
        results["EN"] = EN(image_f)
    if "SD" in metrics:
        results["SD"] = SD(image_f)
    if "SF" in metrics:
        results["SF"] = SF(image_f)
    if "AG" in metrics:
        results["AG"] = AG(image_f)
    if "MI" in metrics:
        results["MI"] = MI(image_f, image_ir, image_vis)
    if "MSE" in metrics:
        results["MSE"] = MSE(image_f, image_ir, image_vis)
    if "CC" in metrics:
        results["CC"] = CC(image_f, image_ir, image_vis)
    if "PSNR" in metrics:
        results["PSNR"] = PSNR(image_f, image_ir, image_vis)
    if "SCD" in metrics:
        results["SCD"] = SCD(image_f, image_ir, image_vis)
    if "VIF" in metrics:
        results["VIF"] = VIFF(image_f, image_ir, image_vis)
    if "Qabf" in metrics:
        results["Qabf"] = Qabf(image_f, image_ir, image_vis)
    if "SSIM" in metrics:
        results["SSIM"] = SSIM(image_f, image_ir, image_vis)

    return results


def evaluate_fast_metric_torch(
    image_f, image_ir, image_vis, metrics: "list[str] | str" = "all"
) -> dict:
    # shapes are [c, h, w], channel is 1 or 3
    # image_f: 0-255

    if "all" == metrics:
        metrics = [
            "EN",
            "SD",
            "SF",
            "AG",
            "MI",
            "MSE",
            "CC",
            "PSNR",
            "SCD",
            "VIF",
            "Qabf",
            "SSIM",
        ]

    results = {}
    if "EN" in metrics:
        results["EN"] = EN_torch(image_f).item()
    if "SD" in metrics:
        results["SD"] = SD_torch(image_f).item()
    if "SF" in metrics:
        results["SF"] = SF_torch(image_f).item()
    if "AG" in metrics:
        results["AG"] = AG_torch(image_f).item()
    if "MI" in metrics:
        results["MI"] = MI_torch(image_f, image_ir, image_vis).item()
    if "MSE" in metrics:
        results["MSE"] = MSE_torch(image_f, image_ir, image_vis).item()
    if "CC" in metrics:
        results["CC"] = CC_torch(image_f, image_ir, image_vis).item()
    if "PSNR" in metrics:
        results["PSNR"] = PSNR_torch(image_f, image_ir, image_vis).item()
    if "SCD" in metrics:
        results["SCD"] = SCD_torch(image_f, image_ir, image_vis).item()
    if "VIF" in metrics:
        results["VIF"] = VIFF_torch(image_f, image_ir, image_vis).item()
    if "Qabf" in metrics:
        results["Qabf"] = Qabf_torch(image_f, image_ir, image_vis).item()
    if "SSIM" in metrics:
        results["SSIM"] = SSIM_torch(image_f, image_ir, image_vis).item()

    return results

if __name__ == "__main__":
    from torchvision.io import read_image
    
    # check shape
    # fused = torch.rand(2, 3, 256, 256)
    # gt = torch.rand(2, 4, 256, 256)
    fused = read_image('/Data3/cao/ZiHanCao/exps/panformer/visualized_img/panRWKV_v8_cond_norm/msrs_v2/00004N.png')[None].float()
    ir = read_image("/Data3/cao/ZiHanCao/datasets/MSRS/test/ir/00004N.jpg")[None].float()
    vi = read_image("/Data3/cao/ZiHanCao/datasets/MSRS/test/vi/00004N.jpg")[None].float()
    
    fused = fused.cuda()[0, 0]
    ir = ir.cuda()[0, 0]
    vi = vi.cuda()[0, 0]
    
    from timeit import timeit
    
    print(timeit(lambda: evaluate_fast_metric_torch(fused, ir, vi, "all"), number=10))
    print(timeit(lambda: evaluate_fast_metric_numpy(fused.cpu().numpy(), ir.cpu().numpy(), vi.cpu().numpy(), "all"), number=10))
