import torch
from torch import nn
from model.module.attention import CAttention, LocalAttention


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, CAttention):
            attn = LocalAttention(dim=m.dim, num_heads=m.num_heads, bias=m.bias, attn_drop=m.attn_drop_prob,
                                  base_size=base_size, fast_imp=False,
                                  train_size=train_size)
            setattr(model, n, attn)


if __name__ == '__main__':
    from model.panformer import PanFormerUNet2

    panformer = PanFormerUNet2(8, 128)
    replace_layers(panformer, 64, 16, False)

    print(panformer)
