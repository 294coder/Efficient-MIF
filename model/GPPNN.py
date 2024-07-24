# -*- coding: utf-8 -*-

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from torch.autograd import Variable
from torch.nn import init

import sys
sys.path.append("./")
from model.base_model import BaseModel, register_model

###############################
######### other blocks ########
###############################


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def uniform_binning_correction(x, n_bits=8):
    """Replaces x^i with q^i(x) = U(x, x + 1.0 / 256.0).

    Args:
        x: 4-D Tensor of shape (NCHW)
        n_bits: optional.
    Returns:
        x: x ~ U(x, x + 1.0 / 256)
        objective: Equivalent to -q(x)*log(q(x)).
    """
    b, c, h, w = x.size()
    n_bins = 2 ** n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return x, objective


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        # return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
        return tensor[:, :1, ...], tensor[:,1:, ...]
    elif type == "cross":
        # return tensor[:, 0::2, ...], tensor[:, 1::2, ...]
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]



def gaussian_p(mean, logs, x):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = logs ** 2
    """
    c = math.log(2 * math.pi)
    return -0.5 * (logs * 2.0 + ((x - mean) ** 2) / torch.exp(logs * 2.0) + c)


def gaussian_likelihood(mean, logs, x):
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])


def gaussian_sample(mean, logs, temperature=1):
    # Sample from Gaussian with temperature
    z = torch.normal(mean, torch.exp(logs) * temperature)

    return z


def squeeze2d(input, factor):
    if factor == 1:
        return input

    B, C, H, W = input.size()

    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x


def unsqueeze2d(input, factor):
    if factor == 1:
        return input

    factor2 = factor ** 2

    B, C, H, W = input.size()

    assert C % (factor2) == 0, "C module factor squared is not 0"

    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)

    return x


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.0):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = -torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited = True

    def _center(self, input, reverse=False):
        if reverse:
            return input - self.bias
        else:
            return input + self.bias

    def _scale(self, input, logdet=None, reverse=False):

        if reverse:
            input = input * torch.exp(-self.logs)
        else:
            input = input * torch.exp(self.logs)

        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply by number of pixels
            """
            b, c, h, w = input.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        self._check_input_dim(input)

        if not self.inited:
            self.initialize_parameters(input)

        if reverse:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        else:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)

        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.0):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()
            )
        )


class LinearZeros(nn.Module):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.logscale_factor = logscale_factor

        self.logs = nn.Parameter(torch.zeros(out_channels))

    def forward(self, input):
        output = self.linear(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        do_actnorm=True,
        weight_std=0.05,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=(not do_actnorm),
        )

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = self.conv(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        logscale_factor=3,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        self.indices = torch.arange(self.num_channels - 1, -1, -1, dtype=torch.long)
        self.indices_inverse = torch.zeros((self.num_channels), dtype=torch.long)

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        shuffle_idx = torch.randperm(self.indices.shape[0])
        self.indices = self.indices[shuffle_idx]

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4

        if not reverse:
            input = input[:, self.indices, :, :]
            return input
        else:
            return input[:, self.indices_inverse, :, :]


class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def split2d_prior(self, z):
        h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = gaussian_sample(mean, logs, temperature)
            z = torch.cat((z1, z2), dim=1)
            return z, logdet
        else:
            z1, z2 = split_feature(input, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = gaussian_likelihood(mean, logs, z2) + logdet
            return z1, logdet


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            output = unsqueeze2d(input, self.factor)
        else:
            output = squeeze2d(input, self.factor)

        return output, logdet


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.linalg.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.process = nn.Sequential(
                nn.Conv2d(channel, channel, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x
        
        

class Refine(nn.Module):

    def __init__(self,in_channels,panchannels,n_feat):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
             CALayer(n_feat,4),
             CALayer(n_feat,4),
             CALayer(n_feat,4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=in_channels-panchannels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out


class Refine1(nn.Module):

    def __init__(self,in_channels,panchannels,n_feat):
        super(Refine1, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
             # CALayer(n_feat,4),
             # CALayer(n_feat,4),
             CALayer(n_feat,4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=in_channels-panchannels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out





########################################################
############## just set in one file ####################
########################################################

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):

    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x

def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)

    return x

def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class PyramidAttention(nn.Module):
    def __init__(self, res_scale=1, channel=64, channels =64, reduction=2, kernelsize=1, ksize=3, stride=1, softmax_scale=10, conv=default_conv):
        super(PyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.escape_NaN = torch.FloatTensor([1e-4])
        self.conv_match_L_base = BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match = BasicBlock(conv,channels,channel//reduction,1,bn=False,act=nn.PReLU())
        self.conv_assembly = BasicBlock(conv, channels, channel, 1, bn=False, act=nn.PReLU())



    def forward(self, input_l, input_s):
        res = input_l
        N, C_l, H_l, W_l = input_l.size()#n,16,h,w
        N, C_s, H_s, W_s = input_s.size()#n,12,h,w
        batch_size = N

        #theta
        match_base = self.conv_match_L_base(input_l)#n,8,h,w
        shape_base = list(res.size())#n,16,h,w
        input_groups = torch.split(match_base, 1, dim=0)

        #patch size for matching
        kernel = self.ksize #3

        #raw_w is for reconstruction
        raw_w = []
        #w is for matching
        w = []

        #conv_f -input:input_s
        base = self.conv_assembly(input_s)
        shape_input = base.shape
        #sampling
        raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
                                        strides=[self.stride, self.stride],
                                        rates=[1, 1],
                                        padding='same') #N, C*k*k, L
        raw_w_i = raw_w_i.view(batch_size, shape_input[1], kernel, kernel, H_s * W_s)
        raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3) #raw_shape:[N,L,C,k,k]
        raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
        raw_w.append(raw_w_i_groups)

        #conv_g -input:input_s
        ref_i = self.conv_match(input_s)
        shape_ref = ref_i.shape
        #sampling
        w_i = extract_image_patches(ref_i, ksizes=[kernel, kernel],
                                    strides=[self.stride, self.stride],
                                    rates=[1, 1],
                                    padding='same')
        w_i = w_i.view(shape_ref[0], shape_ref[1], kernel, kernel, -1)
        w_i = w_i.permute(0, 4, 1, 2, 3) #w shape:[N, L, C, k, k]
        w_i_groups = torch.split(w_i, 1, dim=0)
        w.append(w_i_groups)

        y = []
        for idx, xi in enumerate(input_groups):
            wi = w[0][idx][0] #H_s*W_s, channels//reduction,3,3  64,32,3,3
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                           axis=[1, 2, 3],
                                           keepdim=True)) + self.escape_NaN
            wi_normed = wi/max_wi
            #matching
            xi = same_padding(xi, [kernel, kernel], [1, 1], [1, 1])
            yi = F.conv2d(xi, wi_normed, stride=1)
            yi = yi.view(1, wi.shape[0], shape_base[2], shape_base[3])

            #softmax matching score
            yi = F.softmax(yi*self.softmax_scale, dim=1)

            raw_wi = raw_w[0][idx][0]
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride, padding=1)/4.
            y.append(yi)

        y = torch.cat(y, dim=0) + res*self.res_scale
        return y

######################################################################################################################################################################################################################
def get_kernel():
    k = np.float32([.0625, .25, .375, .25, .0625])
    k = np.outer(k, k)
    return k

#自定义卷积核进行卷积操作
class GaussianPyramid(nn.Module):
    def __init__(self, n):
        super(GaussianPyramid, self).__init__()
        kernel=get_kernel()
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0) #(H,W,1,1)
        self.n=n
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
    def forward(self, img):
        levels=[]
        levels.append(img)
        low=img
        for i in range(self.n - 1):
            low = F.conv2d(low, self.weight, stride=(2,2),padding=(2,2))
            levels.append(low)
        return levels[::-1]
    #return 高斯金字塔


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()

        x_phi = self.conv_phi(x).view(b, c, -1)
        x_theta = self.conv_theta(x).view(b, c, -1)
        x_g = self.conv_g(x).view(b, c, -1)


        x_phi1 = x_phi.view(b, -1)
        x_phi1 = self.softmax(x_phi1)
        x_phi1 = x_phi1.view(b, c, -1)#(N,C,HW)

        x_g1 = x_g.view(b, -1)
        x_g1 = self.softmax(x_g1)
        x_g1 = x_g1.view(b, c, -1) #(N,C,HW)

        y = torch.matmul(x_g1, x_phi1.permute(0, 2, 1).contiguous()) #(N,C,C)
        y = torch.matmul(y, x_theta) #(N, C, HW)

        F_s = y.view(b, self.inter_channel, h, w)
        spatial_out = self.conv_mask(F_s)
        return spatial_out


class subnet(nn.Module):
    def __init__(self, in_channel, num, nums, kernelsize=3, Cross=False):
        super(subnet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channel, out_channels=num, kernel_size=kernelsize, padding=1)
        self.non_local1 = NonLocalBlock(num)
        self.conv1 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=kernelsize, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=kernelsize, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=kernelsize, dilation=3, padding=3)
        self.pyramid = PyramidAttention(channel=num, channels=nums)
        self.conv4 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=kernelsize, dilation=3, padding=3)
        self.conv5 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=kernelsize, dilation=2, padding=2)
        self.conv6 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=kernelsize, padding=1)
        self.non_local2 = NonLocalBlock(num)
        self.recon = nn.Conv2d(in_channels=num, out_channels=8, kernel_size=kernelsize, padding=1)
        self.cross = Cross

    def forward(self, MS, PAN, small=None):
        images = torch.cat((MS,PAN),1)
        x0 = self.conv0(images)
        x0 = x0 + self.non_local1(x0)
        x1 = F.relu(self.conv1(x0))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        middle = x3
        if self.cross:
            middle = self.pyramid(middle, small)

        x4 = F.relu(self.conv4(middle+x3))
        x5 = F.relu(self.conv4(x4+x2))
        x6 = F.relu(self.conv4(x5 + x1))
        x7 = x6 + self.non_local2(x6)
        x7 = MS + self.recon(x7+x0)
        out = x7
        return out, middle, x7

class LPNet(nn.Module):
    def __init__(self, num1, num2, num3):
        super(LPNet, self).__init__()
        self.subnet1 = subnet(in_channel=9, num=num1, nums=num1, Cross=False)
        self.subnet2 = subnet(in_channel=9, num=num2, nums = num1, Cross=True)
        self.subnet3 = subnet(in_channel=9, num=num3, nums = num2, Cross=True)
        self.gaussian = GaussianPyramid(3)

    def forward(self, MS, PAN):
        pyramid = self.gaussian(PAN)
        out1, global_1, outf1 = self.subnet1(MS,pyramid[0])
        out1_t = F.interpolate(outf1, size=[(pyramid[1].shape)[2], (pyramid[1].shape)[3]], mode='nearest')
        out2, global_2, outf2 = self.subnet2(out1_t, pyramid[1], small=global_1)
        out2_t = F.interpolate(outf2, size=[(pyramid[2].shape)[2], (pyramid[2].shape)[3]], mode='nearest')
        out3,_,_ = self.subnet3(out2_t, pyramid[2], small=global_2)
        output_pyramid = []
        output_pyramid.append(out1)
        output_pyramid.append(out2)
        output_pyramid.append(out3)
        return output_pyramid

     ############################################test###############################################################

if __name__ == "__main__":
    
    MS = torch.ones([16, 8, 4, 4])
    PAN = torch.ones([16, 1, 16, 16])
    lpnet = LPNet(12, 16, 24)
    y = lpnet(MS, PAN) #list
    print(type(y), len(y))
    print(y[2].shape)


# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################
class Transformer_Fusion(nn.Module):
    def __init__(self,nc):
        super(Transformer_Fusion, self).__init__()
        self.conv_trans = nn.Sequential(
            nn.Conv2d(2*nc,nc,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(nc,nc,kernel_size=3,stride=1,padding=1))

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lrsr_lv3, ref_lv3):
        ######################   search
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]

        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) #[N, Hr*Wr, H*W]
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1) #[N, H*W]

        ### transfer
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3, 3), padding=1) / (3.*3.)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

        res = self.conv_trans(torch.cat([T_lv3,lrsr_lv3],1))*S+lrsr_lv3

        return res
#
class PatchFusion(nn.Module):
    def __init__(self,nc):
        super(PatchFusion, self).__init__()
        self.fuse = Transformer_Fusion(nc)

    def forward(self,msf,panf):
        ori = msf
        b,c,h,w = ori.size()
        msf = F.unfold(msf,kernel_size=(24, 24), stride=8, padding=8)
        panf = F.unfold(panf, kernel_size=(24, 24), stride=8, padding=8)
        msf = msf.view(-1,c,24,24)
        panf = panf.view(-1,c,24,24)
        fusef = self.fuse(msf,panf)
        fusef = fusef.view(b,c*24*24,-1)
        fusef = F.fold(fusef, output_size=ori.size()[-2:], kernel_size=(24, 24), stride=8, padding=8)
        return fusef

#########################################################################################



def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, channel_out)
        # self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1,self.conv2], 0.1)
        else:
            initialize_weights([self.conv1,self.conv2], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x2


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None

    return constructor


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class FeatureExtract(nn.Module):
    def __init__(self, channel_in=3, channel_split_num=3, subnet_constructor=subnet('DBNet'), block_num=5):
        super(FeatureExtract, self).__init__()
        operations = []

        # current_channel = channel_in
        channel_num = channel_in

        for j in range(block_num):
            b = InvBlock(subnet_constructor, channel_num, channel_split_num)  # one block is one flow step.
            operations.append(b)

        self.operations = nn.ModuleList(operations)
        self.fuse = nn.Conv2d((block_num - 1) * channel_in, channel_in, 1, 1, 0)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x, rev=False):
        out = x  # x: [N,3,H,W]
        outfuse = out
        for i, op in enumerate(self.operations):
            out = op.forward(out, rev)
            if i > 1:
                outfuse = torch.cat([outfuse, out], 1)
        outfuse = self.fuse(outfuse)

        return outfuse


def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)


class Conv_Fusion(nn.Module):
    def __init__(self,nc_in,nc_out):
        super(Conv_Fusion, self).__init__()
        self.conv = nn.Conv2d(nc_in*2,nc_out,3,1,1)
    def forward(self,pan,ms):
        return self.conv(torch.cat([ms,pan],1))

class Conv_Process(nn.Module):
    def __init__(self,ms_channels,pan_channels,nc):
        super(Conv_Process, self).__init__()
        self.convms = nn.Conv2d(ms_channels,nc,3,1,1)
        self.convpan = nn.Conv2d(pan_channels, nc, 3, 1, 1)

    def forward(self,pan,ms):
        return self.convpan(pan),self.convms(ms)


@register_model('gppnn')
class GPPNN(BaseModel):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat,
                 n_layer):
        super(GPPNN, self).__init__()
        self.conv_process = Conv_Process(ms_channels,pan_channels,n_feat//2)
        self.conv_fusion = Conv_Fusion(n_feat//2,n_feat//2)

        self.transform_fusion = PatchFusion(n_feat//2)
        self.extract = FeatureExtract(n_feat, n_feat//2,block_num=3)
        self.refine = Refine1(ms_channels + pan_channels, pan_channels, n_feat)

    def _forward_implem(self, ms, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        mHR = upsample(ms, M, N)
        # finput = torch.cat([pan, mHR], dim=1)
        panf,mHRf = self.conv_process(pan,mHR)
        conv_f = self.conv_fusion(panf,mHRf)
        transform_f = self.transform_fusion(mHRf,panf)
        f_cat = torch.cat([conv_f,transform_f],1)
        # f_cat = conv_f
        fmid = self.extract(f_cat)
        HR = self.refine(fmid)+mHR

        return HR
    
    def train_step(self, ms, lms, pan, gt, criterion):
        fuse = self._forward_implem(ms, pan)
        loss = criterion(fuse, gt)
        return fuse, loss
    
    def val_step(self, ms, lms, pan) -> torch.Tensor:
        return self._forward_implem(ms, pan)
    
    
    
    
if __name__ == '__main__':
    from functools import partial
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    
    torch.cuda.set_device(1)
    
    net = GPPNN(8, 1, 32, 8).cuda()
    
    ms = torch.randn(4, 8, 64, 64).cuda()
    pan = torch.randn(4, 1, 256, 256).cuda()
    
    # print(net(ms, pan).shape)
    
    def forward(self, *args, **kwargs):
        return self._forward_implem(*args, **kwargs)
    
    net.forward = partial(forward, net)
    
    print(
        flop_count_table(FlopCountAnalysis(net, (ms, pan)))
    )
    
    
    

    

