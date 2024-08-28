from typing import Iterable, Optional, Union
import weakref
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)
import accelerate
import torch_ema
import deepspeed
from deepspeed.runtime.zero import GatheredParameters

import sys
sys.path.append('./')
from utils.misc import is_main_process
from utils.deepspeed_ema import DeepspeedEMA
# from utils.utils_modules import Adam_mini


class IdentityScheduler(torch.optim.lr_scheduler._LRScheduler):
    # a placeholder for lr_scheduler or weight_decay_scheduler
    def __init__(self, optim, **kwargs):
        self.optim = optim
        self.kwargs = kwargs

    def step(self, *args, **kwargs):
        pass

    def state_dict(self):
        return self.kwargs
    
    def load_state_dict(self, state_dict):
        pass

def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    """
    copy from DINO. manually set learning lr every iteration.
    note that there is only half epoch of cosine, which means learning rate will not
    go back to the original.
    :param base_value:
    :param final_value:
    :param epochs:
    :param niter_per_ep:
    :param warmup_epochs:
    :param start_warmup_value:
    :return:
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class LinearWarmupScheduler:
    def __init__(self, opt: optim.Optimizer, init_value, warmup_value, warmup_epochs):
        self.opt = opt
        self.init_value = init_value
        self.warmup_value = warmup_value
        self.warmup_epochs = warmup_epochs
        self.values = np.linspace(init_value, warmup_value, warmup_epochs)
        self.now_index = 0

    def step(self):
        self.opt.param_groups[0]["lr"] = self.values[self.now_index]
        self.now_index += 1
        
        
class CosineAnnealingWarmRestartsReduce(CosineAnnealingWarmRestarts):
    def __init__(self, opt: optim.Optimizer, T_0, T_mult=1, lr_mult=1, eta_min=0, last_epoch=-1):
        self.opt = opt
        self.lr_mult = lr_mult
        super().__init__(opt, T_0, T_mult, eta_min, last_epoch)

    def step(self, epoch=None):
        super().step(epoch)
        
        if self.T_cur == self.T_i-1 and self.last_epoch != 0:
            # reduce the base lr
            for i in range(len(self.base_lrs)):
                self.base_lrs[i] *= self.lr_mult
                self.base_lrs[i] = max(self.base_lrs[i], self.eta_min)
                
                
def get_precision(mixed_precision):
    if mixed_precision == 'fp32' or mixed_precision == 'no':
        return torch.float32
    elif mixed_precision == 'fp16':
        return torch.float16
    elif mixed_precision == 'bf16':
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid mixed precision value: {mixed_precision}")
                

def get_scheduler(optim, **kwargs):
    """
    get lr_scheduler or weight_decay_scheduler
    Args:
        optim: optimizer
        **kwargs: a dict containing type of scheduler and its arguments

    Returns: a scheduler

    """
    name = kwargs["name"]
    kwargs.pop("name")
    if name == "cos_anneal":
        return CosineAnnealingLR(optim, **kwargs)
    elif name == "cos_anneal_restart":
        return CosineAnnealingWarmRestarts(optim, **kwargs)
    elif name == "cos_anneal_restart_reduce":
        return CosineAnnealingWarmRestartsReduce(optim, **kwargs)
    elif name == "multi_step":
        return MultiStepLR(optim, **kwargs)
    elif name == "plateau":
        return ReduceLROnPlateau(optim, **kwargs)
    elif name == "identity":
        return IdentityScheduler(optim, **kwargs)
    else:
        raise NotImplementedError


def get_optimizer(model: torch.nn.Module, params: "Iterable | dict", **kwargs):
    name = kwargs["name"]
    kwargs.pop("name")
    if name == "sgd":
        return optim.SGD(params, **kwargs)
    elif name == "adam":
        return optim.Adam(params, **kwargs)
    elif name == "adamw":
        return optim.AdamW(params, **kwargs)
    elif name == 'lion':
        from lion_pytorch import Lion
        return Lion(params, betas=(0.95, 0.98), use_triton=True, **kwargs) 
    elif name == 'fusedadam':
        return deepspeed.ops.adam.FusedAdam(params, **kwargs)
    elif name == 'schedulefree-adam':
        import schedulefree
        return schedulefree.AdamWScheduleFree(params, **kwargs)
    elif name == 'adam-mini':
        return Adam_mini(model, **kwargs)
    else:
        raise NotImplementedError(f'optimizer {name} not implemented')
    
def get_ema_model(parameters: Iterable[torch.nn.Parameter],
                  accelerator: accelerate.Accelerator=None,
                  **ema_kwargs):
    if accelerator is not None:
        if accelerator.state.deepspeed_plugin is not None:
            ema_model = DeepspeedEMA(parameters, **ema_kwargs)
            return ema_model
        
    ema_model = torch_ema.ema.ExponentialMovingAverage(model, **ema_kwargs)
    return ema_model


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch.optim as optim
    import torch.nn as nn

    # init_lr = 1e-3
    # final_lr = 1e-6
    # epochs = 500
    # # nither_per_ep = int(np.ceil(3000 // 16))  # len(datasets) / batch_size
    # # warm_epochs = 80
    # # start_warmup_value = init_lr
    # # cos_sche = cosine_scheduler(
    # #     init_lr, final_lr, epochs, nither_per_ep, warm_epochs, start_warmup_value
    # # )
    # # plt.plot(list(map(lambda x: x / nither_per_ep, range(len(cos_sche)))), cos_sche)
    # # plt.show()

    # # torch cosine annealing lr scheduler
    # net = nn.Sequential(nn.Linear(8, 64))
    # optimizer = optim.AdamW(net.parameters(), lr=init_lr)
    # # cos_sche2 = CosineAnnealingLR(optimizer, epochs - warm_epochs, final_lr)
    # cos_anneal_reduce_sche = CosineAnnealingWarmRestartsReduce(optimizer, 50, 2, 0.5, 1e-6, last_epoch=-1)
    
    # lr = []
    # for i in range(200, 500):
    #     l = optimizer.param_groups[0]["lr"]
    #     lr.append(l)
    #     # if i > warm_epochs:
    #     #     cos_sche2.step()
    #     cos_anneal_reduce_sche.step(i)
    # plt.plot(range(200, 500), lr)
    # # plt.show()
    # plt.savefig('cos_anneal_reduce.png')
    
    import accelerate
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    
    accelerator = accelerate.Accelerator()
    model = torch.nn.Linear(3, 16)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(*([torch.randn(16, 3)]*2)), batch_size=1)
    model, dl, opt = accelerator.prepare(model, dl, opt)
    
    ema_model = get_ema_model(list(model.parameters()), accelerator=accelerator, decay=0.9)
    
    accelerator.wait_for_everyone()
    
    if is_main_process():
        print('main process: set weight to zero weight \n')
        model.weight.data.zero_()
        model.bias.data.zero_()
        print(model.weight)
    else:
        import time
        time.sleep(2)
        # the other process
        print('other process: set weight to non-zero weight\n')
        print(ema_model.shadow_params)
        
    accelerator.wait_for_everyone()
    print('-----------------------------'*2, '\n')
    ema_model.update()
    
    # proc 0 is zeros but proc 1 is not,
    # so the ema_model should not be all zeros
    if is_main_process():
        print('main process')
        print(ema_model.shadow_params)
        print(model.weight)
        print('-----------------------------'*2, '\n')
        ema_model.restore(model.parameters())
        print(model.weight)
        print('-----------------------------'*2)
        
    
    
    
