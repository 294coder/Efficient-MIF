from functools import partial
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.module import PatchMergeModule


## TODO: old import fasion, now we directly import from the model path
# register all model name in a global dict
MODELS = {}

# use it in a decorator way
# e.g.
# @register_model('model_name')
def register_model(name):
    def inner(cls):
        MODELS[name] = cls
        return cls

    return inner


# base model class
# all model defination should inherit this class
from abc import ABC, abstractmethod
class BaseModel(ABC, nn.Module):
    
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not (cls._is_method_implemented('train_step') or cls._is_method_implemented('fusion_train_step')):
            raise NotImplementedError(f"{cls.__name__} must implement at least one of the methods: 'train_step' or 'fusion_train_step'")

        if not (cls._is_method_implemented('val_step') or cls._is_method_implemented('fusion_val_step')):
            raise NotImplementedError(f"{cls.__name__} must implement at least one of the methods: 'val_step' or 'fusion_val_step'")

    @staticmethod
    def _is_method_implemented(method):
        return any(method in B.__dict__ for B in BaseModel.__subclasses__())
    
    def train_step(
        self, ms, lms, pan, gt, criterion
    ) -> tuple[torch.Tensor, tuple[Tensor, dict[str, Tensor]]]:
        raise NotImplementedError

    def val_step(self, ms, lms, pan) -> torch.Tensor:
        raise NotImplementedError
    
    def fusion_train_step(self, vis, ir, mask, gt, criterion) -> tuple[torch.Tensor, tuple[Tensor, dict[str, Tensor]]]:
        raise NotImplementedError
    
    def fusion_val_step(self, vis, ir, mask) -> torch.Tensor:
        raise NotImplementedError

    def patch_merge_step(self, *args) -> torch.Tensor:
        # not compulsory
        raise NotImplementedError

    def forward(self, *args, mode="train", **kwargs):
        if mode == "train":
            return self.train_step(*args, **kwargs)
        elif mode == "eval":
            return self.val_step(*args, **kwargs)
        elif mode == 'fusion_train':
            return self.fusion_train_step(*args, **kwargs)
        elif mode == 'fusion_eval':
            return self.fusion_val_step(*args, **kwargs)
        elif mode == "patch_merge":
            raise DeprecationWarning("patch_merge is deprecated.")
            # return self.patch_merge_step(*args, **kwargs)
        else:
            raise NotImplementedError

    @abstractmethod
    def _forward_implem(self, *args, **kwargs):
        raise NotImplementedError
