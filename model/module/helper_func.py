from typing import Sequence
from einops import rearrange
import torch.nn as nn


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def empty(x):
    return x.numel() == 0


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


class MultiInputSequential(nn.Sequential):
    def forward(self, *input):
        for module in self:
            if isinstance(input, Sequence):
                input = module(*input)
            else:
                input = module(input)
        return input
