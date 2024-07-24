from typing import Iterable, Optional, Union
import weakref
import torch
import torch.nn as nn
import copy
import contextlib
import deepspeed
from deepspeed.runtime.zero import GatheredParameters

from utils.misc import is_main_process


class DeepspeedEMA:
    def __init__(self, 
                parameters: Iterable[torch.nn.Parameter],
                decay: float,
                use_num_updates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.decay = decay
        self.m_name2s_name = {}
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = []
        self.collected_params = []
        self._params_refs = []
        
        with GatheredParameters(list(parameters)) and torch.no_grad():  # no need to register to extern parameters
            for p in parameters:
                if p.requires_grad:
                    # weakref still work?
                    # FIXME: may cause OOM on one gpu?
                    self._params_refs.append(weakref.ref(p))
                    self.shadow_params.append(nn.Parameter(p.detach().clone()))
                    
    def _get_parameters(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]]
    ) -> Iterable[torch.nn.Parameter]:
        if parameters is None:
            parameters = [p() for p in self._params_refs]
            if any(p is None for p in parameters):
                raise ValueError(
                    "(One of) the parameters with which this "
                    "ExponentialMovingAverage "
                    "was initialized no longer exists (was garbage collected);"
                    " please either provide `parameters` explicitly or keep "
                    "the model to which they belong from being garbage "
                    "collected."
                )
            return parameters
        else:
            parameters = list(parameters)
            if len(parameters) != len(self.shadow_params):
                raise ValueError(
                    "Number of parameters passed as argument is different "
                    "from number of shadow parameters maintained by this "
                    "ExponentialMovingAverage"
                )
            return parameters
        
    def update(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(
                decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )
        one_minus_decay = 1.0 - decay
        
        with GatheredParameters(parameters, 0) and torch.no_grad():
            if is_main_process():
                for s_param, param in zip(self.shadow_params, parameters):
                    param = param.type_as(s_param)
                    tmp = (s_param - param)
                    # tmp will be a new tensor so we can do in-place
                    tmp.mul_(one_minus_decay)
                    s_param.sub_(tmp)

    # def forward(self, model):
    #     decay = self.decay

    #     if self.num_updates >= 0:
    #         self.num_updates += 1
    #         decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    #     one_minus_decay = 1.0 - decay
    #     shadow_params = dict(self.named_buffers())

    #     with torch.no_grad():
    #         with GatheredParameters(model.parameters()):
    #             if is_main_process():
    #                 m_param = dict(model.named_parameters())

    #                 for key in m_param:
    #                     if m_param[key].requires_grad:
    #                         sname = self.m_name2s_name[key]
    #                         shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
    #                         shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
    #                     else:
    #                         assert not key in self.m_name2s_name

    def copy_to(
        self, 
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ):
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        with GatheredParameters(parameters, modifier_rank=0) and torch.no_grad():
            if is_main_process():
                for s_param, param in zip(self.shadow_params, parameters):
                    param.data.copy_(s_param.data)

    def store(
        self, 
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ):
        """
        Save the current parameters for restoring later.
        Args:
          model: A model that parameters will be stored
        """
        with GatheredParameters(parameters) and torch.no_grad():
            if is_main_process():
                self.collected_params = [param.detach().clone() for param in parameters]

    def restore(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        if self.collected_params is None:
            raise RuntimeError(
                "This ExponentialMovingAverage has no `store()`ed weights "
                "to `restore()`"
            )
        parameters = self._get_parameters(parameters)
        with GatheredParameters(parameters, modifier_rank=0):
            if is_main_process():
                for c_param, param in zip(self.collected_params, parameters):
                    param.data.copy_(c_param.data)
                    
    @contextlib.contextmanager
    def average_parameters(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ):
        r"""
        Context manager for validation/inference with averaged parameters.

        Equivalent to:

            ema.store()
            ema.copy_to()
            try:
                ...
            finally:
                ema.restore()

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)
    
    def state_dict(self) -> dict:
        r"""Returns the state of the ExponentialMovingAverage as a dict."""
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the ExponentialMovingAverage state.

        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.num_updates = state_dict["num_updates"]
        assert self.num_updates is None or isinstance(self.num_updates, int), \
            "Invalid num_updates"

        self.shadow_params = state_dict["shadow_params"]
        assert isinstance(self.shadow_params, list), \
            "shadow_params must be a list"
        assert all(
            isinstance(p, torch.Tensor) for p in self.shadow_params
        ), "shadow_params must all be Tensors"

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            assert isinstance(self.collected_params, list), \
                "collected_params must be a list"
            assert all(
                isinstance(p, torch.Tensor) for p in self.collected_params
            ), "collected_params must all be Tensors"
            assert len(self.collected_params) == len(self.shadow_params), \
                "collected_params and shadow_params had different lengths"

        if len(self.shadow_params) == len(self._params_refs):
            # Consistant with torch.optim.Optimizer, cast things to consistant
            # device and dtype with the parameters
            params = [p() for p in self._params_refs]
            # If parameters have been garbage collected, just load the state
            # we were given without change.
            if not any(p is None for p in params):
                # ^ parameter references are still good
                for i, p in enumerate(params):
                    self.shadow_params[i] = self.shadow_params[i].to(
                        device=p.device, dtype=p.dtype
                    )
                    if self.collected_params is not None:
                        self.collected_params[i] = self.collected_params[i].to(
                            device=p.device, dtype=p.dtype
                        )
        else:
            raise ValueError(
                "Tried to `load_state_dict()` with the wrong number of "
                "parameters in the saved state."
            )
