import math
from typing import Union
from copy import deepcopy
from bytecode import Bytecode, Instr
from accelerate import Accelerator
from accelerate.utils import DistributedType

import torch
import torch.nn as nn


def hook_model(model: nn.Module, saved_tensor, hook_class):
    def feature_hook(_, input, output):
        # forward hook
        saved_tensor.append([input, output])

    hooks = []
    for n, m in model.named_modules():
        if isinstance(m, hook_class):
            hooks.append(m.register_forward_hook(feature_hook))
    return model, hooks


class loss_with_l2_regularization(nn.Module):
    def __init__(self):
        super(loss_with_l2_regularization, self).__init__()

    def forward(self, loss, model, weight_decay=1e-5, flag=False):
        regularizations = []
        for k, v in model.named_parameters():
            if "conv" in k and "weight" in k:
                # print(k)
                penality = weight_decay * ((v.data ** 2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))
        # r = torch.sum(regularizations)

        loss = loss + sum(regularizations)
        return loss


def variance_scaling_initializer(tensor):
    # stole it from woo-xiao.
    # thanks
    def calculate_fan(shape, factor=2.0, mode="FAN_IN", uniform=False):
        # 64 9 3 3 -> 3 3 9 64
        # 64 64 3 3 -> 3 3 64 64
        if shape:
            # fan_in = float(shape[1]) if len(shape) > 1 else float(shape[0])
            # fan_out = float(shape[0])
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])
        else:
            fan_in = 1.0
            fan_out = 1.0
        for dim in shape[:-2]:
            fan_in *= float(dim)
            fan_out *= float(dim)
        if mode == "FAN_IN":
            # Count only number of input connections.
            n = fan_in
        elif mode == "FAN_OUT":
            # Count only number of output connections.
            n = fan_out
        elif mode == "FAN_AVG":
            # Average number of inputs and output connections.
            n = (fan_in + fan_out) / 2.0
        if uniform:
            raise NotImplemented
            # # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
            # limit = math.sqrt(3.0 * factor / n)
            # return random_ops.random_uniform(shape, -limit, limit,
            #                                  dtype, seed=seed)
        else:
            # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
            trunc_stddev = math.sqrt(1.3 * factor / n)
        return fan_in, fan_out, trunc_stddev
    
def model_params(model: nn.Module, accelerator=None):
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            return get_fp32_state_dict_from_zero_checkpoint(model)
        else:
            return model.state_dict()
        
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    elif isinstance(model, torch._dynamo.eval_frame.OptimizedModule):  # torch.compile model
        model = model._orig_mod
    return model.state_dict()

def model_device(model: Union[nn.Module, nn.DataParallel, 
                              nn.parallel.DistributedDataParallel,
                              torch._dynamo.eval_frame.OptimizedModule]):
    params = model.parameters()
    p0 = next(params)
    return p0.device


def clip_norm_(max_norm, network, fp_scaler=None, optim=None, accelerator=None):
    if fp_scaler is not None:
        fp_scaler.unscale_(optim)
    if accelerator is not None:
        accelerator.clip_grad_norm_(network.parameters(), max_norm)
    else:
        torch.nn.utils.clip_grad.clip_grad_norm_(network.parameters(), max_norm)
    
def clip_value_(max_value, network, fp_scaler=None, optim=None, accelerator=None):
    if fp_scaler is not None:
        fp_scaler.unscale_(optim)
    if accelerator is not None:
        accelerator.clip_grad_value(network.parameters(), max_value)
    else:
        torch.nn.utils.clip_grad.clip_grad_value_(network.parameters(), max_value)


def step_loss_backward(
        optim,
        network=None,
        max_norm=None,
        max_value=None,
        loss=None,
        fp16=False,  # decrepted
        mix_precision=False,
        fp_scaler=None,
        grad_accum=False,
        accelerator: Accelerator=None,
):
    """

    :param optim: optimizer. type: optim.Optimizer
    :param network: instanced network. type: nn.Module
    :param max_norm: clip norm. type: float
    :param loss: float
    :param fp16: bool
    :param fp_scaler: mix-precision scaler
    :return:
    """
    if (fp16 and fp_scaler is None) or (not fp16 and fp_scaler is not None):
        if accelerator is None:
            raise ValueError("fp16 and grad_scaler should be set together")
        else:
            fp16 = False
    if max_norm is not None and network is None:
        raise ValueError("max_norm is set, network should be set")
    
    mixed_precision = fp16
    
    optim.zero_grad()
    if mixed_precision:
        fp_scaler.scale(loss).backward()
        if max_norm is not None:
            clip_norm_(max_norm, network, fp_scaler, optim)
        if not grad_accum:
            fp_scaler.step(optim)
            fp_scaler.update()
    else:
        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)
        
        # assert max_norm and max_value can not be set at the same time
        if max_norm is not None:
            clip_norm_(max_norm, network, accelerator=accelerator)
        elif max_value is not None:
            clip_value_(network.parameters(), max_value, accelerator=accelerator)
                
        if not grad_accum:
            optim.step()
            
def find_no_or_big_norm_params(net: nn.Module, ktop: int=20):
    """find unused params and big-normed gradient

    Args:
        net (nn.Module): network to be checked.
        ktop (int, optional): top k params to be printed. Defaults to 20.
    """
    # find unused params and big-normed gradient
    d_grads = {}
    n_params = 0
    for n, p in net.named_parameters():
        n_params += p.numel()
        if p.grad is None:
            print(n, "has no grad")
        else:
            p_sum = torch.abs(p.grad).sum().item()
            d_grads[n] = p_sum

    # topk
    d_grads = dict(sorted(d_grads.items(), key=lambda item: item[1], reverse=True))
    for k, v in list(d_grads.items())[:ktop]:
        print(k, v)


class EMAModel(object):
    def __init__(self, model, ema_ratio=0.9999):
        super().__init__()
        self.model = model
        self.ema_ratio = ema_ratio
        self.ema_model = deepcopy(model)

    def update(self):
        for ema_p, now_p in zip(self.ema_model.state_dict(), self.model.state_dict()):
            ema_p.data = ema_p.data * self.ema_ratio + now_p.data * (1 - self.ema_ratio)

    def ema_model_state_dict(self):
        try:
            return self.ema_model.module.state_dict()
        except:
            return self.ema_model.state_dict()
        
        

class get_local(object):
    cache = {}
    is_activate = False

    def __init__(self, *args):
        self.varname = args

    def __call__(self, func):
        if not type(self).is_activate:
            return func

        type(self).cache[func.__qualname__] = []
        c = Bytecode.from_code(func.__code__)
        extra_code = []
        
        extra_code.extend([
            *[Instr('LOAD_FAST', varn) for varn in self.varname],
            Instr('BUILD_LIST', len(self.varname)),
            Instr('STORE_FAST', '_result_list'),
            Instr('LOAD_FAST', '_result_list'),
            Instr('BUILD_TUPLE', 2)
        ])

        c[-1:-1] = extra_code
        func.__code__ = c.to_code()

        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            output_vs = res[0]
            saved_vs = res[1]
            
            # assume is a Tensor
            for i, v in enumerate(saved_vs):
                if hasattr(v, 'detach'):
                    v = v.detach().cpu()  #.numpy()
                    saved_vs[i] = v
            
            type(self).cache[func.__qualname__].append(saved_vs)
            return output_vs
        return wrapper

    @classmethod
    def clear(cls):
        for key in cls.cache.keys():
            cls.cache[key] = []

    @classmethod
    def activate(cls):
        from loguru import logger
        
        logger.warning('ready to get local varibles, be careful about you CPU memory.')
        cls.is_activate = True
        
# if __name__ == '__main__':

#     get_local.activate()

#     @get_local('x', 'y')
#     def func():
#         x = 1
#         y = [1,2]
        
#         return x, y
        
        
#     print(func())
#     print(get_local.cache)
    