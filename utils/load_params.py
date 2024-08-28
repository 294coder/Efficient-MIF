from functools import partial
from numpy import isin
import torch
import logging
from collections import OrderedDict
from safetensors.torch import load_file
from packaging import version

from utils.log_utils import easy_logger


def module_load(path, 
                model, 
                device, 
                ddp_rank=None, 
                strict=True, 
                spec_key='shadow_params', 
                logger=None, 
                full_unmatched_log=True):
    if logger is None:
        logger = easy_logger(func_name=__name__)
    
    model = model.to(device if ddp_rank is None else ddp_rank)
    place = device if ddp_rank is None else {'cuda:%d' % 0: 'cuda:%d' % ddp_rank}
    if isinstance(place, torch.device):
        place = str(place)
    if path.endswith('pth') or path.endswith('pt'):
        if version.parse(torch.__version__) >= version.parse('2.4.0'):
            load_engine = partial(torch.load, map_location=place, weights_only=False)
        else:
            load_engine = partial(torch.load, map_location=place)
    elif path.endswith('safetensors'):
        load_engine = lambda weight_path, map_location: OrderedDict(load_file(weight_path, device=map_location))
    else:
        raise ValueError
    
    try:
        params = load_engine(path, map_location=place)
    except Exception:
        # TODO: the exception is not used
        logger.print('>>> did not find the pth file, try to find in used_weights/ and ununsed_weights/...', logging.INFO)
        try:
            path_used = path.replace('weight/', 'weight/used_weights/')
            params = load_engine(path_used, map_location=place)
        except Exception:
            path_ununsed = path.replace('weight/', 'weight/unused_weights/')
            params = load_engine(path_ununsed, map_location=place)
        
    
    # parse key
    if spec_key is not None:
        parsed_keys = spec_key.split('.')
        try:
            for k in parsed_keys:
                params = params[k]
        except KeyError:
            logger.warning(f'>>> not found parsed model `{spec_key}`, load the model directly \n \n')
        
    _load_fail_flag = False

    params_load = params
    # may be tedious but useful and safe to avoid 'module.' prefix caused error
    if not strict:
        logger.warning('model load strict is False, set it to True if you know what you are doing')
        
    def _iter_params_load_fn(model, params_load, strict):
        nonlocal _load_fail_flag
        
        if not isinstance(params_load, (list, tuple)):
            param_load_ziped = list(params_load.items()) if isinstance(params_load, dict) else params_load
            for (s_name, s_param), (name, param) in zip(param_load_ziped, model.named_parameters()):
                saved_shape = tuple(s_param.data.shape)
                required_shape = tuple(param.data.shape)
                if saved_shape != required_shape:
                    if strict:
                        logger.print(
                            f'param shape unmatched, {name} requires: {required_shape}, but got {s_name}: {saved_shape}',
                            logging.WARNING
                        )
                        if not full_unmatched_log:
                            logger.print('model load failed! shape of params does not match!', logging.ERROR)
                            raise RuntimeError('model load failed! shape of params does not match!')
                        else:
                            _load_fail_flag = True
                            continue
                    else:
                        logger.print(f'skip the shape mismatched param, param name {name}, '
                                        + f'current shape {required_shape} but loaded shape {saved_shape}', logging.WARNING)
                        continue
                param.data.copy_(s_param.data)
        else:
            for s_param, param in zip(params_load, model.parameters()):
                required_shape = tuple(param.data.shape)
                saved_shape = tuple(s_param.data.shape)
                
                if saved_shape != required_shape:
                    if strict:
                        logger.print(
                            f'param shape unmatched, requires: {required_shape}, but got {saved_shape}',
                            logging.WARNING
                        )
                        if not full_unmatched_log:
                            logger.print('model load failed! shape of params does not match!', logging.ERROR)
                            raise RuntimeError('model load failed! shape of params does not match!')
                        else:
                            _load_fail_flag = True
                            continue
                    else:
                        logger.print(f'skip the shape mismatched param, current shape {required_shape} but loaded shape {saved_shape}', logging.WARNING)
                        continue
                param.data.copy_(s_param.data)
        
    def _load_fn(model, params_load, strict):
        
        if isinstance(params_load, OrderedDict):  # ordered dict
            model.load_state_dict(params_load, strict=strict)
        else:
            _iter_params_load_fn(model, params_load, strict)
        
                
    _load_fn(model, params_load, strict)
    
    if _load_fail_flag:
        raise RuntimeError('model load failed! shape of params does not match!')
    
    
    # except Exception:
    #     # data parallel mode will save params with keys' prefix is 'module'.
    #     odict = {}
    #     for k, v in params_load.items():
    #         odict['module.' + k] = v
    #         params_load[spec_key] = odict
        
    #     if 'ema' not in spec_key:
    #         _load_fn(model, params_load, strict)
    #     else: 
    #         raise RuntimeError('ema model load failed! shape of params does not match!')
            
    logger.print('load pretrain weights', logging.INFO)
    return model


# def resume_load(path,
#                 model,
#                 optim,
#                 lr_scheduler,
#                 ema_model: ExponentialMovingAverage=None,
#                 specific_resume_lr: float = None,
#                 specific_epochs: int = None,
#                 wd_scheduler=None,
#                 device='cuda:0',
#                 ddp_rank=None,
#                 ddp=False):
#     # @specific_resume_lr(warning: not recommended):
#     # manually specify learning rate when the lr from last break is too low to update model

#     # @specific_epochs(warning: not recommended):
#     # manually specify total epochs when resuming training

#     model.to(device if ddp_rank is None else ddp_rank)
#     # assume saved params always on cuda:0
#     params = torch.load(path, map_location=device if ddp_rank is None else {'cuda:%d' % 0: 'cuda:%d' % ddp_rank})

#     # NOTE: ddp mode will save params with keys' prefix is 'module'.
#     #  now I remove the prefix for just one card circumstance but it conflict with ddp mode.
#     if ddp:
#         odict = OrderedDict()
#         for k, v in params['model'].items():
#             odict['module.' + k] = v
#             params['model'] = odict
#     model.load_state_dict(params['model'])
    
#     if ema_model is not None:
#         ema_model.load_state_dict(params['ema_model'])

#     # NOTE: Pytorch 1.12.0 may cause CUDA error in optimizer reloading. see more at
#     # https://github.com/pytorch/pytorch/issues/80809#issuecomment-1175211598
#     optim.load_state_dict(params['optim'])
#     if specific_resume_lr is not None:
#         optim.param_groups[0]['lr'] = specific_resume_lr
        
#     lr_scheduler.load_state_dict(params['lr_scheduler'])
    
#     if specific_epochs is not None:
#         # FIXME: only support CosineAnnealing lr_scheduler
#         lr_scheduler.T_max = specific_epochs
        
#     resume_ep = params['epochs']
#     print(f"last training resume! best metrics are {params['metrics']}")

#     # warning: if you change total epochs in the resume run, the lr_scheduler may not update lr
#     if wd_scheduler is not None:
#         wd_scheduler.load_state_dict(params['wd_scheduler'])
#         return model, optim, lr_scheduler, wd_scheduler, resume_ep
#     else:
#         return model, optim, lr_scheduler, resume_ep
