import math, warnings
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RWKV_TimeMix_x051a(nn.Module):

    def __init__(self, n_embd, n_head, n_layer, dropout, bias, layer_id):
        super().__init__()
        assert n_embd % n_head == 0

        self.head_size = n_embd // n_head
        self.n_head = n_head

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -6 + 5 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.unsqueeze(-1))

            tmp = torch.zeros(self.n_head)
            for h in range(self.n_head):
                tmp[h] = ratio_0_to_1 * (1 - (h / (self.n_head - 1)))
            self.time_faaaa = nn.Parameter(tmp.unsqueeze(-1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = nn.Linear(n_embd, n_embd, bias=bias)
        self.key = nn.Linear(n_embd, n_embd, bias=bias)
        self.value = nn.Linear(n_embd, n_embd, bias=bias)
        self.gate = nn.Linear(n_embd, n_embd, bias=bias)

        self.output = nn.Linear(n_embd, n_embd, bias=bias)
        self.ln_x = nn.GroupNorm(self.n_head, n_embd, eps=(1e-5)*64)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        H, N = self.n_head, self.head_size
        #
        # we divide a block into chunks to speed up computation & save vram.
        # you can try to find the optimal chunk_len for your GPU.
        # avoid going below 128 if you are using bf16 (otherwise time_decay might be less accurate).
        #
        if T % 64 == 0: Q = 64
        # elif T % 32 == 0: Q = 32
        elif T % 128 == 0: Q = 128
        else:
            Q = T
            warnings.warn(f'\n{"#"*80}\n\n{" "*38}Note\nThe GPT-mode forward() should only be called when we are training models.\n' \
                          'Now we are using it for inference for simplicity, which works, but will be very inefficient.\n\n{"#"*80}\n')
        assert T % Q == 0

        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xr = x + xx * self.time_maa_r
        xg = x + xx * self.time_maa_g
        r = self.receptance(xr).view(B, T, H, N).transpose(1, 2) # receptance
        k = self.key(xk).view(B, T, H, N).permute(0, 2, 3, 1) # key
        v = self.value(xv).view(B, T, H, N).transpose(1, 2) # value
        g = F.gelu(self.gate(xg)) # extra gate  # silu here

        w = torch.exp(-torch.exp(self.time_decay.float())) # time_decay
        u = self.time_faaaa.float() # time_first

        ws = w.pow(Q).view(1, H, 1, 1)

        ind = torch.arange(Q-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, Q).pow(ind)

        wk = w.view(1, H, 1, Q)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, Q))
        w = torch.tile(w, [Q])
        w = w[:, :-Q].view(-1, Q, 2*Q - 1)
        w = w[:, :, Q-1:].view(1, H, Q, Q)

        w = w.to(dtype=r.dtype) # the decay matrix
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)

        state = torch.zeros(B, H, N, N, device=r.device, dtype=r.dtype) # state
        y = torch.empty(B, H, T, N, device=r.device, dtype=r.dtype) # output

        for i in range(T // Q): # the rwkv-x051a operator
            # chunk along the length dim
            rr = r[:, :, i*Q:i*Q+Q, :]  # [B, H, Q, N]
            kk = k[:, :, :, i*Q:i*Q+Q]
            vv = v[:, :, i*Q:i*Q+Q, :]
            
            # y1 = (rr @ kk) * w: [B, H, Q, Q]
            # y2 = y1 @ vv: [B, H, Q, N]
            # y3 = (rr @ state) * wb: [B, H, Q, N]
            y[:, :, i*Q:i*Q+Q, :] = ((rr @ kk) * w) @ vv + (rr @ state) * wb
            
            # state: [1, H, 1, 1] * [B, H, N, N] + ([B, H, N, Q] * [1, H, Q, 1]) @ [B, H, Q, N] = [B, H, N, N]
            state = ws * state + (kk * wk) @ vv

        y = y.transpose(1, 2).contiguous().view(B * T, C)
        y = self.ln_x(y).view(B, T, C) * g

        # output projection
        y = self.dropout(self.output(y))
        return y

class RWKV_ChannelMix_x051a(nn.Module):

    def __init__(self, n_embd, bias, drop, n_layer, layer_id):
        super().__init__()

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.value = nn.Linear(3 * n_embd, n_embd, bias=bias)
        self.receptance = nn.Linear(n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        x = self.key(xk)
        x = torch.relu(x) ** 2
        x = self.value(x)
        x = torch.sigmoid(self.receptance(xr)) * x
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, n_embd, n_head, bias, tmix_drop, cmix_drop, n_layer, layer_id):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.tmix = RWKV_TimeMix_x051a(n_embd, n_head, n_layer, tmix_drop, bias, layer_id)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.cmix = RWKV_ChannelMix_x051a(n_embd, bias, cmix_drop, n_layer, layer_id)

    def forward(self, x):
        x = x + self.tmix(self.ln_1(x))
        x = x + self.cmix(self.ln_2(x))
        return x
    
    
if __name__=='__main__':
    import torch
    
    device = torch.device('cuda:0')
    x = torch.randn(1, 32*32, 64).to(device)
    model = Block(64, 8, True, 0.1, 0.1, 10, 0).to(device)
    y = model(x)
    print(y.shape)
    
    # print(torch.cuda.memory_summary())
    