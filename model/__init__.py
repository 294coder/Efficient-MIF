import os
import sys
import importlib
from importlib.util import find_spec, LazyLoader, module_from_spec

sys.path.append(os.path.dirname(__file__))

## torch dynamo setting
import torch
torch.set_float32_matmul_precision('high')
torch._dynamo.config.cache_size_limit = 64
print('torch._dynamo.config.cache_size_limit:', torch._dynamo.config.cache_size_limit)

## suppress warnings
import warnings
warnings.filterwarnings("ignore", module="torch.utils")
warnings.filterwarnings("ignore", module="deepspeed.accelerator")


## register all models
from model.base_model import MODELS, BaseModel
from utils import easy_logger

logger = easy_logger(func_name='model_registry')

__all__ = [
    'MODELS',
    'BaseModel',
    'build_network',
    'lazy_load_network'
]

_all_modules = [
    
]

_all_model_class_name = [
    
]

assert len(_all_modules) == len(_all_model_class_name), 'length of modules and registry names should be the same'

_module_network_dict = {k: v for k, v in zip(_all_modules, _all_model_class_name)}


def _lazy_load_module():
    lazy_loader_dict = {}
    for module_name, class_name in _module_network_dict.items():
        spec = find_spec(module_name)
        if spec is not None:
            spec.loader = LazyLoader(spec.loader)
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            lazy_loader_dict[f'{module_name}.{class_name}'] = module
        else:
            raise ValueError(f'Module: {module_name} not found')
    return lazy_loader_dict
            
LAZY_LOADER_DICT = _lazy_load_module()

def _active_load_module(module_name, model_class_name):
    dict_name = f'{module_name}.{model_class_name}'
    if dict_name in LAZY_LOADER_DICT:
        logger.info(f'loading {dict_name}')
        module = LAZY_LOADER_DICT[dict_name]
        return getattr(module, model_class_name)
    else:
        logger.critical(f'{dict_name} not found in LAZY_LOADER_DICT')
        raise ValueError(f'{dict_name} not found in LAZY_LOADER_DICT')


# TODO: hydra import
# ==============================================
# register all models
from .LEMamba import LEMambaNet
















# ==============================================



## FIXME: may cause conficts with other arguments in args that rely on static registered model name in main.py
def import_model_from_name(name):
    module = importlib.import_module(name, package='model')
    model_cls = getattr(module, name)
    return model_cls

def build_network(model_name: str=None, **kwargs) -> BaseModel:
    """
    build network from model name and kwargs
    
    """
    
    assert model_name is not None, 'model_name is not specified'
    try:
        net = MODELS.get(model_name)
    except:
        try:
            net = import_model_from_name(model_name)
        except:
            net = MODELS.get(model_name.split('.')[-1])
        
    assert net is not None, f'no model named {model_name} is registered'
    
    return net(**kwargs)

def lazy_load_network(module_name: str=None,
                      model_class_name:str=None, **kwargs) -> BaseModel:
    """
    lazy load model from module_name and registry_model_name
    
    """
    
    assert module_name is not None, 'module_name is not specified'
    assert model_class_name is not None,'model_class_name is not specified'
    
    return _active_load_module(module_name, model_class_name)(**kwargs)


