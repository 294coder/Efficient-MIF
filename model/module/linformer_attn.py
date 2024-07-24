# mainly modified from
# https://github.com/lucidrains/linformer/blob/master/linformer/linformer.py
import math
import torch
from torch import nn


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class LinformerCrossAttention(nn.Module):
    def __init__(self,
                 dim1,
                 dim2,
                 ) -> None:
        super().__init__()
        


class LinformerSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        num_feats=256,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        share_kv=False,
    ):
        super().__init__()
        assert (
            dim % num_heads
        ) == 0, "dimension must be divisible by the number of heads"

        self.seq_len = seq_len
        self.num_feats = num_feats

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, num_feats)))
        if share_kv:
            self.proj_v = self.proj_k
        else:
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, num_feats)))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, d = x.shape
        d_h, h, k = self.head_dim, self.num_heads, self.num_feats
        kv_len = n
        assert (
            kv_len == self.seq_len
        ), f"the sequence length of the key / values must be {self.seq_len} - {kv_len} given"

        queries = self.scale * self.query(x).reshape(b, n, h, d_h).transpose(
            1, 2
        )  # [b, h, n, d]
        kv = self.kv(x).reshape(b, n, 2, d).permute(2, 0, 1, 3)  # [2, b, n, d]
        keys, values = (
            kv[0],
            kv[1],
        )  # make torchscript happy (cannot use tensor as tuple)

        # project keys and values along the sequence length dimension to k
        # complexity: O(d*n*k)
        proj_seq_len = lambda args: torch.einsum(
            "bnd,nk->bkd", *args
        )  # [b, n, d] @ [n, k] -> [b, k, d]
        kv_projs = (self.proj_k, self.proj_v)
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values
        merge_key_values = (
            lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        )
        keys, values = map(merge_key_values, (keys, values))  # [b, h, k, d]

        # attention
        # complexity: O(n*d*k)
        attn = torch.einsum("bhnd,bhkd->bhnk", queries, keys)  # [b, h, n, k]
        attn = (attn - torch.max(attn, dim=-1, keepdim=True)[0]).softmax(dim=-1)
        attn = self.attn_drop(attn)
        # complexity: O(n*k*d)
        out = torch.einsum("bhnk,bhkd->bhnd", attn, values)  # [b, h, n, d]

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


if __name__ == "__main__":
    x = torch.randn(1, 256, 128)  # [b, l, d]
    lin_sa = LinformerSelfAttention(128, 256)
    print(lin_sa(x).shape)

