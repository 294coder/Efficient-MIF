import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
import sys
sys.path.append('./')

from model.base_model import register_model, BaseModel


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Resblock(nn.Module):
    def __init__(self, channel=32, ksize=3, padding=1):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=ksize,
            padding=padding,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=ksize,
            padding=padding,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv1(x))  # Bsx32x64x64
        rs1 = self.conv2(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


class LayerNorm2d(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)


class ReflashValue(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim * 2, 1, bias=True),
        )

    def forward(self, v, v2):
        scale, shift = self.body(v2).chunk(2, dim=1)
        return v * (1 + scale) + shift


class ReflashAttn(nn.Module):
    def __init__(self, nhead=8, ksize=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                nhead, nhead, (1, ksize), stride=1, padding=(0, ksize // 2), bias=False
            ),
            nn.Conv2d(nhead, nhead, 1, bias=True),
            nn.ReLU(),
        )
        self.body[0].weight.data.fill_(1.0 / (ksize * ksize))
        self.body[1].weight.data.fill_(1.0)
        self.body[1].bias.data.fill_(0.0)

    def forward(self, attn):
        return self.body(attn).softmax(-1)
    
    
def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)
    # windows = rearrange(x, "b c (h w1) (w w2) -> (b w1 w2) c h w", h=window_size, w=window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    # x = rearrange(windows, '(b w1 w2) c h w -> b c (h w1) (w w2)', 
    #               b=B, w1=H//window_size, w2=W//window_size)
    return x


# TODO:
# 1. 不加resblock换v
# 2. 加上resblok不换v


class FirstAttn(nn.Module):
    def __init__(
        self, pan_dim, lms_dim, inner_dim, nheads=8, attn_drop=0.2, first_layer=False,
        window_size = 8,
    ) -> None:
        super().__init__()
        self.nheads = nheads
        self.first_layer = first_layer
        
        self.ws = window_size

        # self.rearrange = Rearrange("b (nhead c) h w -> b nhead c (h w)", nhead=nheads)
        self.q = nn.Sequential(
            nn.Conv2d(pan_dim, inner_dim, 1, bias=True),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, groups=inner_dim),
            # nn.Linear(pan_dim, inner_dim)
        )
        self.k = nn.Sequential(
            nn.Conv2d(lms_dim, inner_dim, 1, bias=True),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, groups=inner_dim),
            # nn.Linear(lms_dim, inner_dim)
        )
        self.v = nn.Sequential(
            nn.Conv2d(lms_dim, inner_dim, 1, bias=True),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, groups=inner_dim),
            # nn.Linear(lms_dim, inner_dim)
        )
        self.proj_out = nn.Linear(inner_dim, inner_dim)
        if first_layer:
            self.q.append(
                nn.Sequential(
                    Resblock(inner_dim),
                    Resblock(inner_dim),
                    Resblock(inner_dim),
                    Resblock(inner_dim),
                )
            )
            self.kv.append(
                nn.Sequential(
                    Resblock(inner_dim * 2),
                    Resblock(inner_dim * 2),
                    Resblock(inner_dim * 2),
                    Resblock(inner_dim * 2),
                )
            )

        self.ms_pre_norm = LayerNorm2d(lms_dim)
        self.pan_pre_norm = LayerNorm2d(pan_dim)

        self.attn_drop = nn.Dropout(attn_drop)

    # @get_local("out")
    def forward(self, lms, pan):
        *_, h, w = lms.shape

        lms = self.ms_pre_norm(lms)
        pan = self.pan_pre_norm(pan)
        
        
        q = self.q(pan)  # q: pan
        k = self.k(lms)
        v = self.v(lms)  # kv: lms
        q, k, v = map(lambda x: window_partition(x, self.ws), (q, k, v))
        q, k, v= map(lambda x: rearrange(x, 'b wh ww (head c) -> b head (wh ww) c', head=self.nheads), 
                     [q, k, v])
        attn = q @ k.transpose(-2, -1)  # [b, head, n, c] @ [b, head, c, m]
        attn = self.attn_drop(attn)
        attn = attn.softmax(-1)
        out = attn @ v  # [b, head, n, m] @ [b, head, m, c]
        out = rearrange(out, "b nhead n d -> b n (nhead d)")
        out = self.proj_out(out).transpose(-2, -1).contiguous()
        
        out = window_reverse(out, self.ws, h, w)
        return attn, out


class MSReversibleRefine(nn.Module):
    def __init__(self, dim, hp_dim, nhead=8, first_stage=False, window_size=8) -> None:
        super().__init__()
        self.ws = window_size
        self.res_block = Residual(
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 1),
            )
        )
        if not first_stage:
            self.proj_v = nn.Linear(dim, dim)
            self.reflash_attn = ReflashAttn(nhead=nhead)
        self.fuse_conv = nn.Conv2d(dim + hp_dim, dim, 3, 1, 1)

        self.nhead = nhead
        self.first_stage = first_stage

    @get_local("reflashed_out")
    def forward(self, reuse_attn, refined_lms, hp_in):
        *_, h, w = refined_lms.shape

        if not self.first_stage:
            reuse_attn = self.reflash_attn(reuse_attn)  # [b, h, m, n]
            refined_lms = window_partition(refined_lms, self.ws)  # [b, wh, ww, c]
            refined_lms = self.proj_v(refined_lms)
            refined_lms = rearrange(refined_lms, "b h w (nhead c) -> b nhead (h w) c", nhead=self.nhead)
            refined_lms = reuse_attn @ refined_lms
            refined_lms = rearrange(
                refined_lms,
                "b nhead (wh ww) c -> b (nhead c) wh ww",
                nhead=self.nhead,
                wh=self.ws,
                ww=self.ws,
            )
            refined_lms = window_reverse(refined_lms, self.ws, h, w)
            
        reflashed_attn = reuse_attn
        reflashed_out = refined_lms
        refined_lms = self.res_block(refined_lms)
        reverse_out = torch.cat([refined_lms, hp_in], dim=1)
        out = self.fuse_conv(reverse_out)
        out = reflashed_out * out

        return reuse_attn, out


class PreHp(nn.Module):
    def __init__(self, pan_dim, lms_dim, hp_dim) -> None:
        super().__init__()
        self.hp_pan = nn.Sequential(
            nn.Conv2d(pan_dim, pan_dim, 3, 1, 1, groups=pan_dim),
            nn.Conv2d(pan_dim, hp_dim, 1),
        )
        self.hp_lms = nn.Sequential(
            nn.Conv2d(lms_dim, lms_dim, 3, 1, 1, groups=lms_dim),
            nn.Conv2d(lms_dim, hp_dim, 1),
        )
        # laplacian init
        self.hp_pan[0].weight.data = torch.tensor(
            [[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32
        ).repeat(pan_dim, 1, 1, 1)
        self.hp_pan[0].bias.data = torch.tensor([0.0] * pan_dim, dtype=torch.float32)
        self.hp_lms[0].weight.data = torch.tensor(
            [[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32
        ).repeat(lms_dim, 1, 1, 1)
        self.hp_lms[0].bias.data = torch.tensor([0.0] * lms_dim, dtype=torch.float32)

        self.hp_conv = nn.Conv2d(hp_dim * 2, hp_dim, 1)

        self.msrb = nn.Sequential(
            Resblock(hp_dim), Resblock(hp_dim), Resblock(hp_dim), Resblock(hp_dim)
        )

    def forward(self, lms, pan):
        hp_pan = self.hp_pan(pan)
        hp_lms = self.hp_lms(lms)
        x = torch.cat([hp_pan, hp_lms], dim=1)
        hp_in = self.msrb(self.hp_conv(x))
        return hp_in


class HpBranch(nn.Module):
    def __init__(self, attn_dim, hp_dim, n=2) -> None:
        super().__init__()
        self.attn_hp_conv = nn.Conv2d(attn_dim, hp_dim, 1)
        self.to_hp_dim = nn.Conv2d(hp_dim * 2, hp_dim, 1)
        self.res_block = nn.Sequential(*[Resblock(hp_dim) for _ in range(n)])

    def forward(self, refined_lms, hp_in):
        attn_hp = self.attn_hp_conv(refined_lms)
        x = torch.cat([attn_hp, hp_in], dim=1)
        hp_out = self.res_block(self.to_hp_dim(x))
        return hp_out


@register_model("lformer")
class AttnFuseMain(BaseModel):
    def __init__(
        self,
        pan_dim,
        lms_dim,
        attn_dim,
        hp_dim,
        n_stage=3,
        patch_merge=True,
        crop_batch_size=1,
        patch_size_list=None,
        scale=4,
    ) -> None:
        super().__init__()
        self.n_stage = n_stage

        self.attn = FirstAttn(pan_dim, lms_dim, attn_dim, first_layer=False)
        self.pre_hp = PreHp(pan_dim, lms_dim, hp_dim)

        self.refined_blocks = nn.ModuleList([])
        self.hp_branch = nn.ModuleList([])
        for i in range(n_stage):
            b = MSReversibleRefine(attn_dim, hp_dim, first_stage=(i == 0))
            self.refined_blocks.append(b)

        for i in range(n_stage + 1):
            b = HpBranch(attn_dim, hp_dim)
            self.hp_branch.append(b)

        # self.final_conv = nn.Conv2d(hp_dim, lms_dim, 1)
        self.final_conv = nn.Sequential(
            Resblock(hp_dim + attn_dim),
            Resblock(hp_dim + attn_dim),
            nn.Conv2d(hp_dim + attn_dim, lms_dim, 1),
        )

        self.patch_merge = patch_merge
        if patch_merge:
            from model.base_model import PatchMergeModule

            self._patch_merge_model = PatchMergeModule(
                patch_merge_step=self.patch_merge_step,
                crop_batch_size=crop_batch_size,
                patch_size_list=patch_size_list,
                scale=scale,
            )

    def _forward_implem(self, lms, pan):
        reused_attn, refined_lms = self.attn(lms, pan)
        pre_hp = self.pre_hp(lms, pan)
        for i in range(self.n_stage):
            reversed_out = self.hp_branch[i](refined_lms, pre_hp)
            # TODO: update reused_attn or not
            reused_attn, refined_lms = self.refined_blocks[i](
                reused_attn, refined_lms, reversed_out
            )

        out = self.hp_branch[-1](refined_lms, pre_hp)
        out = torch.cat([out, refined_lms], dim=1)

        out = self.final_conv(out)

        return out

    def train_step(self, ms, lms, pan, gt, criterion):
        pred = self._forward_implem(lms, pan)

        out = pred + lms
        loss = criterion(out, gt)
        return out.clip(0, 1), loss

    def val_step(self, ms, lms, pan, patch_merge=True):
        if self.patch_merge and patch_merge:
            pred = self._patch_merge_model.forward_chop(ms, lms, pan)[0]
        else:
            pred = self._forward_implem(lms, pan)
        out = pred + lms

        return out.clip(0, 1)

    def patch_merge_step(self, ms, lms, pan):
        return self._forward_implem(lms, pan)
