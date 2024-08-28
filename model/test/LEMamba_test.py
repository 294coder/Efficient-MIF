import pytest
from warnings import filterwarnings
filterwarnings("ignore")
from rich.console import Console
console = Console()

import torch
from model.LEMamba import LEMambaNet



img_channel = 8
condition_channel = 1
out_channel = 8
width = 24
middle_blk_nums = 1

naf_enc_blk_nums = []
naf_dec_blk_nums = []
naf_chan_upscale = []

ssm_enc_blk_nums = [2, 1, 1]
ssm_dec_blk_nums = [2, 1, 1]
ssm_chan_upscale = [1, 1, 1]
ssm_ratios = [2, 2, 1]
window_sizes = [8, 8, None]
ssm_enc_d_states = [[16, 32], [16, 32], [None, 32]]
ssm_dec_d_states = [[None, 32], [16, 32], [16, 32]]
ssm_enc_convs = [[7, 11], [7, 11], [None, 11]]
ssm_dec_convs = [[None, 11], [7, 11], [7, 11]]

pt_img_size = 64
if_rope = False
if_abs_pos = False
patch_merge = True
upscale = 4
device = 'cuda'


@pytest.mark.parametrize(
    'ssm_enc_blk_nums, middle_blk_nums, ssm_dec_blk_nums, ssm_enc_convs, ssm_dec_convs, ssm_ratios, ssm_chan_upscale, ssm_enc_d_states, ssm_dec_d_states, window_sizes, device',
    [
        (ssm_enc_blk_nums, middle_blk_nums, ssm_dec_blk_nums, ssm_enc_convs, ssm_dec_convs, ssm_ratios, ssm_chan_upscale, ssm_enc_d_states, ssm_dec_d_states, window_sizes, device)
    ]
)
def test_model_forward(ssm_enc_blk_nums,
                       middle_blk_nums,
                       ssm_dec_blk_nums,
                       ssm_enc_convs,
                       ssm_dec_convs,
                       ssm_ratios,
                       ssm_chan_upscale,
                       ssm_enc_d_states,
                       ssm_dec_d_states,
                       window_sizes,
                       device):

    model = LEMambaNet(
        img_channel=img_channel,
        condition_channel=condition_channel,
        out_channel=out_channel,
        width=width,
        middle_blk_nums=middle_blk_nums,
        naf_enc_blk_nums=naf_enc_blk_nums,
        naf_dec_blk_nums=naf_dec_blk_nums,
        naf_chan_upscale=naf_chan_upscale,
        ssm_enc_blk_nums=ssm_enc_blk_nums,
        ssm_dec_blk_nums=ssm_dec_blk_nums,
        ssm_chan_upscale=ssm_chan_upscale,
        ssm_ratios=ssm_ratios,
        window_sizes=window_sizes,
        ssm_enc_d_states=ssm_enc_d_states,
        ssm_dec_d_states=ssm_dec_d_states,
        ssm_enc_convs=ssm_enc_convs,
        ssm_dec_convs=ssm_dec_convs,
        pt_img_size=pt_img_size,
        if_rope=if_rope,
        if_abs_pos=if_abs_pos,
        patch_merge=patch_merge,
        upscale=upscale
    ).to(device)
    
    scale = 4
    img_sz = pt_img_size
    ms_img_sz = img_sz // scale
    device = 'cuda'
    chan = img_channel
    pan_chan = condition_channel
    ms = torch.randn(1, chan, ms_img_sz, ms_img_sz).to(device)
    img = torch.randn(1, chan, ms_img_sz * scale, ms_img_sz * scale).to(device)
    cond = torch.randn(1, pan_chan, ms_img_sz * scale, ms_img_sz * scale).to(device)
    gt = torch.randn(1, chan, ms_img_sz * scale, ms_img_sz * scale).to(device)

    output = model._forward_implem(img, cond)
    output.sum().backward()
    
    assert output.shape == (1, out_channel, pt_img_size, pt_img_size)
    assert next(model.parameters()).grad is not None
