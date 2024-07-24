import numpy as np
import pandas as pd
import os
import random
import time
import cv2
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F


def high_level(hs, high_hs, batch_size, band_hs, win_size, h, w):
    index_map = torch.argmax(high_hs, dim=1)  # N H W
    index_map = index_map.reshape([batch_size, -1])  # N H*W
    # min_index = torch.min(index_map)
    # index_map = rescale_value * (index_map - min_index) / (torch.max(index_map) - min_index)

    sort_index = torch.argsort(index_map, dim=1)

    hs0 = hs.reshape([batch_size, band_hs, -1]) #.numpy()
    sorted_hs = []
    for i in range(batch_size):
        # print(hs0[i, :, sort_index[i, :]][None, ...].shape)
        sorted_hs.append(
            # torch.Tensor(
                hs0[i, :, sort_index[i, :]].T[None, ...]
            # )
        )

    sorted_hs = torch.concat(sorted_hs, 0)
    sorted_hs = sorted_hs.reshape(
        [batch_size, band_hs, win_size, int(h * w / win_size)]
    )
    std_sorted_hs = torch.std(sorted_hs, dim=2)
    return torch.log(torch.sum(std_sorted_hs))


def gaussian1d(window_size, sigma):
    ###window_size = 11
    x = torch.arange(window_size)
    x = x - window_size // 2
    gauss = torch.exp(-(x**2) / float(2 * sigma**2))
    # print('gauss.size():', gauss.size())
    ### torch.Size([11])
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian1d(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    # print('2d',_2D_window.shape)
    # print(window_size, sigma, channel)
    return _2D_window.expand([channel, 1, window_size, window_size])


def _ssim(
    img1,
    img2,
    window,
    window_size,
    channel=126,
    data_range=255.0,
    size_average=True,
    C=None,
):
    # size_average for different channel

    padding = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
    # print(mu1.shape)
    # print(mu1[0,0])
    # print(mu1.mean())
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
    if C == None:
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
    else:
        C1 = (C[0] * data_range) ** 2
        C2 = (C[1] * data_range) ** 2
    # l = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    sc = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    lsc = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * sc

    if size_average:
        ### ssim_map.mean()是对这个tensor里面的所有的数值求平均
        return lsc.mean()
    else:
        # ## 返回各个channel的值
        return lsc.flatten(2).mean(-1), sc.flatten(2).mean(-1)


class SSIMLoss(torch.nn.Module):
    """
    1. 继承paddle.nn.Layer
    """

    def __init__(self, window_size=11, channel=3, data_range=255.0, sigma=1.5):
        """
        2. 构造函数根据自己的实际算法需求和使用需求进行参数定义即可
        """
        super(SSIMLoss, self).__init__()
        self.data_range = data_range
        self.C = [0.01, 0.03]
        self.window_size = window_size
        self.channel = channel
        self.sigma = sigma
        window = create_window(self.window_size, self.sigma, self.channel)
        self.register_buffer("window", window)
        # print(self.window_size,self.window.shape)

    def forward(self, input, label):
        """
        3. 实现forward函数，forward在调用时会传递两个参数：input和label
            - input：单个或批次训练数据经过模型前向计算输出结果
            - label：单个或批次训练数据对应的标签数据
            接口返回值是一个Tensor，根据自定义的逻辑加和或计算均值后的损失
        """
        # 使用Paddle中相关API自定义的计算逻辑
        # output = xxxxx
        # return output
        return 1 - _ssim(
            input,
            label,
            data_range=self.data_range,
            window=self.window,
            window_size=self.window_size,
            channel=self.channel,
            size_average=True,
            C=self.C,
        )


# 通用函数


def pdfFilesPath(path):
    """
    path: 目录文件夹地址
    返回值：列表，pdf文件全路径
    """
    filePaths = []  # 存储目录下的所有文件名，含路径
    for root, dirs, files in os.walk(path):
        for file in files:
            filePaths.append(os.path.join(root, file))
    return filePaths


def read_srf(filename, skiprows=11):
    data = np.loadtxt(filename, skiprows=skiprows)
    return data


def read_band(filename):  # 读取wv2影像波段和波宽
    data = open(filename, "r")
    f = data.read()

    band = str.split(f, ",")
    band = [float(x) for x in band]
    length = len(np.array(band))
    band_center = np.array(band)[: int(length / 2)]
    band_width = np.array(band)[int(length / 2) :]

    if band_center[0] > 100:
        return band_center, band_width
    else:
        return band_center * 1000, band_width * 1000


def generate_srf(filename4):  # 产生和每个高光谱波段
    srf = read_srf(filename4[0])
    srf[:, 0] = srf[:, 0] * 1000
    srf = srf[:, 0:7]  # 放弃海岸线和近红波段

    band_center, band_width = read_band(filename4[1])
    left_band = band_center - band_width / 2.0
    right_band = band_center + band_width / 2.0

    srf_simu = np.expand_dims(np.zeros(6), axis=0)
    for i0 in range(len(band_center)):
        index = np.where((srf[:, 0] >= left_band[i0]) & (srf[:, 0] <= right_band[i0]))
        srf_simu = np.concatenate((srf_simu, np.mean(srf[index, 1:], axis=1)), axis=0)

    band_center = np.expand_dims(band_center, axis=0)
    print(srf_simu.shape)
    srf_simu = np.concatenate((np.transpose(band_center), srf_simu[1:, :]), axis=1)

    return srf_simu


def all_valid(data, axis=0):
    # a, b, c = data.shape
    # 波段轴在1
    # if a < b and a < c:
    if axis == 0:
        length = data.shape[1] * data.shape[2]
    elif axis == 2:
        length = data.shape[0] * data.shape[1]
    sum_data = np.sum(data, axis=axis)
    if len(np.where(sum_data > 0)[0]) == length:
        return True
    else:
        return False


def blur_downsampling(original_msi, ratio=4, band_index=0):
    if ratio >= 9:
        kernel_size = ratio
    else:
        kernel_size = 9

    # kernel_size = 9   # =ratio
    # kernel_size = ratio

    """generating image with gaussian kernel"""
    sig = (1 / (2 * 2.7725887 / ratio**2)) ** 0.5
    kernel = np.multiply(
        cv2.getGaussianKernel(kernel_size, sig),
        cv2.getGaussianKernel(kernel_size, sig).T,
    )
    new_lrhs2 = []
    for i in range(original_msi.shape[0]):  # every band
        temp = original_msi[i, :, :]
        temp = np.expand_dims(
            signal.convolve2d(temp, kernel, boundary="wrap", mode="same"), axis=0
        )
        new_lrhs2.append(temp)
        print(i)
    new_lrhs2 = np.concatenate(new_lrhs2, axis=0)

    return new_lrhs2[:, int(ratio / 2) :: ratio, int(ratio / 2) :: ratio]


def log(base, x):
    return np.log(x) / np.log(base)


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == "u":
        x.uniform_()

    elif noise_type == "n":
        x.normal_()
    else:
        assert False


def get_noise(shape, method="noise", noise_type="u", var=1.0 / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    # if isinstance(spatial_size, int):
    #     spatial_size = (spatial_size, spatial_size)
    # if method == 'noise':
    # net_input = torch.zeros(shape)

    # fill_noise(net_input, noise_type)
    # net_input *= var
    net_input = torch.uniform(shape, dtype="float32", min=0.0, max=0.1)

    return net_input


class lap_conv(nn.Module):
    def __init__(self, band_ms, ks=3):
        super(lap_conv, self).__init__()
        pass

        # sig = 1.5
        # kernel = np.multiply(cv2.getGaussianKernel(ks, sig),
        #                     cv2.getGaussianKernel(ks, sig).T)
        # self.kernel = torch.to_tensor(np.expand_dims(np.expand_dims(kernel, axis=0), axis=0), dtype='float32')
        #
        # lap_matrix = np.expand_dims(np.expand_dims(np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32), axis=0), axis=0)
        # lap_matrix = np.tile(lap_matrix, (1, band_ms, 1, 1))     # lap matrix
        # weight_attr = torch.ParamAttr(initializer=nn.initializer.Assign(lap_matrix),trainable=False)
        # self.lap_conv = nn.Conv2D(band_ms, band_ms, kernel_size=ks, padding=1, padding_mode='circular', weight_attr=weight_attr)

    def forward(self, x):
        pass


class Downsampler(nn.Module):
    """
    http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    """

    def __init__(
        self,
        factor,
        kernel_size=9,
        padding=0,
        sig=6.794574429554831,
        n_planes=1,
        phase=0,
        kernel_width=None,
        support=None,
        sigma=None,
        preserve_size=False,
    ):
        super(Downsampler, self).__init__()
        self.ratio = factor
        self.padding = padding
        kernel_size = kernel_size

        # sig = (1 / (2 * 2.7725887 / 16 ** 2)) ** 0.5
        kernel = np.multiply(
            cv2.getGaussianKernel(kernel_size, sig),
            cv2.getGaussianKernel(kernel_size, sig).T,
        )
        self.kernel = torch.Tensor(
            np.expand_dims(np.expand_dims(kernel, axis=0), axis=0)
        )

    def forward(self, input0):
        row = input0.shape[2]
        col = input0.shape[3]
        print(row, col)
        # print(self.ratio)
        # print(int(row/self.ratio))
        # print(int(col/self.ratio))
        result = torch.zeros(
            [
                input0.shape[0],
                input0.shape[1],
                int(row / self.ratio),
                int(col / self.ratio),
            ]
        )
        # print(torch.expand(input[0, 0, :, :], [1, 1, row, col]).shape)
        for j in range(input0.shape[0]):
            for i in range(input0.shape[1]):
                result[j, i, :, :] = F.conv2d(
                    torch.reshape(input0[j, i, :, :], [1, 1, row, col]),
                    self.kernel,
                    stride=self.ratio,
                    padding=self.padding,
                )
        return result


def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ["lanczos", "gauss", "box"]

    # factor  = float(factor)
    if phase == 0.5 and kernel_type != "box":
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])

    if kernel_type == "box":
        assert phase == 0.5, "Box filter is always half-phased"
        kernel[:] = 1.0 / (kernel_width * kernel_width)

    elif kernel_type == "gauss":
        assert sigma, "sigma is not specified"
        assert phase != 0.5, "phase 1/2 for gauss not implemented"

        center = (kernel_width + 1.0) / 2.0
        print(center, kernel_width)
        sigma_sq = sigma * sigma

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center) / 2.0
                dj = (j - center) / 2.0
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2.0 * np.pi * sigma_sq)
    elif kernel_type == "lanczos":
        assert support, "support is not specified"
        center = (kernel_width + 1) / 2.0

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor

                pi_sq = np.pi * np.pi

                val = 1
                if di != 0:
                    val = (
                        val
                        * support
                        * np.sin(np.pi * di)
                        * np.sin(np.pi * di / support)
                    )
                    val = val / (np.pi * np.pi * di * di)

                if dj != 0:
                    val = (
                        val
                        * support
                        * np.sin(np.pi * dj)
                        * np.sin(np.pi * dj / support)
                    )
                    val = val / (np.pi * np.pi * dj * dj)

                kernel[i - 1][j - 1] = val

    else:
        assert False, "wrong method name"

    kernel /= kernel.sum()

    return kernel


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        stride=stride,
        bias=bias,
    )


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        bn=False,
        act=nn.PReLU(),
    ):
        m = [default_conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


def normalize(x):
    return x.mul_(2).add_(-1)


def same_padding(images, ksizes, strides, rates):
    assert len(images.shape) == 4
    batch_size, channel, rows, cols = images.shape
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.0)
    padding_left = int(padding_cols / 2.0)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    pad = nn.ConstantPad2d(padding=paddings, value=0)
    images = pad(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding="same"):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.shape) == 4
    assert padding in ["same", "valid"]
    batch_size, channel, height, width = images.shape

    if padding == "same":
        images = same_padding(images, ksizes, strides, rates)
    elif padding == "valid":
        pass
    else:
        raise NotImplementedError(
            'Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(
                padding
            )
        )

    unfold = nn.Unfold(kernel_size=ksizes, dilation=rates, padding=0, stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def bgr2ycbcr(img, only_y=True):
    """bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.0
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img,
            [
                [24.966, 112.0, -18.214],
                [128.553, -74.203, -93.786],
                [65.481, -37.797, 112.0],
            ],
        ) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    """same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.0
    # convert
    rlt = np.matmul(
        img,
        [
            [0.00456621, 0.00456621, 0.00456621],
            [0, -0.00153632, 0.00791071],
            [0.00625893, -0.00318811, 0],
        ],
    ) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument(
        "--opt", help="experiment configure file name", required=True, type=str
    )
    parser.add_argument(
        "--root_path",
        help="experiment configure file name",
        default="../../../",
        type=str,
    )
    # distributed training
    parser.add_argument("--gpu", help="gpu id for multiprocessing training", type=str)
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )

    args = parser.parse_args()

    return args
