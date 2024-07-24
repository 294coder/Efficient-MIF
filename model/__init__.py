# import os
# import importlib

import warnings
warnings.filterwarnings("ignore", module="torch.utils")
warnings.filterwarnings("ignore", module="deepspeed.accelerator")

# from model.build_network import build_network
from model.base_model import MODELS, BaseModel

# TODO: hydra import
# ==============================================
# register all models
from .LEMamba import LEMambaNet


import importlib
import sys
import torch

sys.path.append('./')

# FIXME: may cause conficts with other arguments in args that rely on static registered model name in main.py
def import_model_from_name(name):
    module = importlib.import_module(name, package='model')
    model_cls = getattr(module, name)
    return model_cls

def build_network(model_name:str=None, **kwargs) -> BaseModel:
    assert model_name is not None, 'model_name is not specified'
    try:
        net = MODELS.get(model_name)
    except:
        try:
            net = import_model_from_name(model_name)
        except:
            net = MODELS.get(model_name.split('.')[-1])
        
    assert net is not None, f'no model named {model_name} is registered'
    # import networks
    return net(**kwargs)


import hydra
from omegaconf import OmegaConf

def build_model_from_name(cfg: OmegaConf):
    model = hydra.utils.instantiate(cfg)
    
    return model

# ==============================================
