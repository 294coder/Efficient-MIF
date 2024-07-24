import argparse
from contextlib import contextmanager
import json
from multiprocessing import context
import os
import os.path as osp
import random
import time
from typing import Dict, Iterable, Union
import importlib
import h5py
from fvcore.nn import FlopCountAnalysis, flop_count_table

import yaml
import numpy as np
import torch
import torch.distributed as dist
import kornia
import shortuuid
from matplotlib import pyplot as plt
from torch.backends import cudnn


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def set_all_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def to_numpy(*args):
    l = []
    for i in args:
        if isinstance(i, torch.Tensor):
            l.append(i.detach().cpu().numpy())
    return l


def to_tensor(*args, device, dtype):
    out = []
    for a in args:
        out.append(torch.tensor(a, dtype=dtype).to(device))
    return out


def to_device(*args, device):
    out = []
    for a in args:
        out.append(a.to(device))
    return out

def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image to YCbCr.
    
    Args:
        image: RGB image tensor with shape (..., 3, H, W) in range [0, 1]
    
    Returns:
        YCbCr image tensor with shape (..., 3, H, W)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (..., 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.29900 * r + 0.58700 * g + 0.11400 * b
    cb: torch.Tensor = -0.168736 * r - 0.331264 * g + 0.50000 * b + 0.5
    cr: torch.Tensor = 0.50000 * r - 0.418688 * g - 0.081312 * b + 0.5

    return torch.stack([y, cb, cr], dim=-3)

def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """
    Convert a YCbCr image to RGB.
    
    Args:
        image: YCbCr image tensor with shape (..., 3, H, W)
    
    Returns:
        RGB image tensor with shape (..., 3, H, W) in range [0, 1]
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (..., 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    r: torch.Tensor = y + 1.40200 * (cr - 0.5)
    g: torch.Tensor = y - 0.34414 * (cb - 0.5) - 0.71414 * (cr - 0.5)
    b: torch.Tensor = y + 1.77200 * (cb - 0.5)

    return torch.stack([r, g, b], dim=-3).clamp(0, 1)

@contextmanager
def y_pred_model_colored(vis: torch.Tensor, enable: bool=True):
    """
    Context manager to handle YCbCr color space conversion for image processing.
    
    Args:
    vis (torch.Tensor): Input RGB image tensor of shape (B, 3, H, W)
    
    Yields:
    torch.Tensor: Y channel of the image
    
    The context manager handles:
    1. Converting RGB to YCbCr
    2. Extracting Y channel for processing
    3. Converting processed Y channel back to RGB
    """
    
    if enable:
        assert vis.size(1) == 3, 'vis should be a 3-channel rgb image'
        y_cb_cr = kornia.color.rgb_to_ycbcr(vis)
        cbcr = y_cb_cr[:, 1:]
        y = y_cb_cr[:, :1]
        
        def back_to_rgb(pred_y):
            y_cb_cr = torch.cat([pred_y, cbcr], dim=1)
            return kornia.color.ycbcr_to_rgb(y_cb_cr)
    else:
        y = vis
        def back_to_rgb(pred_rgb):
            return pred_rgb
    
    try:
        # Yield the Y channel for processing
        yield y, back_to_rgb
            
    finally:
        pass
        

def h5py_to_dict(file: h5py.File, keys=None) -> dict[str, np.ndarray]:
    """get all content in a h5py file into a dict contains key and values

    Args:
        file (h5py.File): h5py file
        keys (list, optional): h5py file keys used to extract values.
        Defaults to ["ms", "lms", "pan", "gt"].

    Returns:
        dict[str, np.ndarray]:
    """
    d = {}
    if keys is None:
        keys = list(file.keys())
    for k in keys:
        v = file[k][:]
        d[k] = v
    return d


def dict_to_str(d, decimals=4):
    n = len(d)
    func = lambda k, v: f"{k}: {torch.round(v, decimals=decimals).item() if isinstance(v, torch.Tensor) else np.round(v, decimals=decimals)}"
    s = ""
    for i, (k, v) in enumerate(d.items()):
        s += func(k, v) + (", " if i < n - 1 else "")
    return s


def prefixed_dict_key(d, prefix, sep="_"):
    # e.g.
    # SSIM -> train_SSIM
    d2 = {}
    for k, v in d.items():
        d2[prefix + sep + k] = v
    return d2


# TODO: nees test
class CheckPointManager(object):
    def __init__(
        self,
        model: torch.nn.Module,
        save_path: str,
        save_every_eval: bool = False,
        verbose: bool = True,
    ):
        """
        manage model checkpoints
        Args:
            model: nn.Module, can be single node model or multi-nodes model
            save_path: str like '/home/model_ckpt/resnet.pth' or '/home/model_ckpt/exp1' when @save_every_eval
                       is False or True
            save_every_eval: when False, save params only when ep_loss is less than optim_loss.
                            when True, save params every eval epoch
            verbose: print out all information

        e.g.
        @save_every_eval=False, @save_path='/home/ckpt/resnet.pth'
        weights will be saved like
        -------------
        /home/ckpt
        |-resnet.pth
        -------------

        @save_every_eval=True, @save_path='/home/ckpt/resnet'
        weights will be saved like
        -------------
        /home/ckpt
        |-resnet
            |-ep_20.pth
            |-ep_40.pth
        -------------

        """
        self.model = model
        self.save_path = save_path
        self.save_every_eval = save_every_eval
        self._optim_loss = torch.inf
        self.verbose = verbose

        self.check_path_legal()

    def check_path_legal(self):
        if self.save_every_eval:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        else:
            assert self.save_path.endswith(".pth")
            par_dir = os.path.dirname(self.save_path)
            if not os.path.exists(par_dir):
                os.makedirs(par_dir)

    def save(
        self,
        ep_loss: Union[float, torch.Tensor] = None,
        ep: int = None,
        extra_saved_dict: dict = None,
    ):
        """

        Args:
            ep_loss: should be set when @save_every_eval=False
            ep: should be set when @save_every_eval=True
            extra_saved_dict: a dict which contains other information you want to save with model
                            e.g. {'optimizer_ckpt': op_ckpt, 'time': '2023/1/21'}

        Returns:

        """
        if isinstance(ep_loss, torch.Tensor):
            ep_loss = ep_loss.item()

        saved_dict = {}
        if not self.save_every_eval:
            assert ep_loss is not None
            if ep_loss < self._optim_loss:
                self._optim_loss = ep_loss
                path = self.save_path
                saved_dict["optim_loss"] = ep_loss
            else:
                print(
                    "optim loss: {}, now loss: {}, not saved".format(
                        self._optim_loss, ep_loss
                    )
                )
                return
        else:
            assert ep is not None
            path = os.path.join(self.save_path, "ep_{}.pth".format(ep))

        if extra_saved_dict is not None:
            assert "model" not in list(saved_dict.keys())
            saved_dict = extra_saved_dict

        try:
            saved_dict["model"] = self.model.module.state_dict()
        except:
            saved_dict["model"] = self.model.state_dict()

        torch.save(saved_dict, path)

        if self.verbose:
            print(
                f"saved params contains\n",
                *[
                    "\t -{}: {}\n".format(k, v if k != "model" else "model params")
                    for k, v in saved_dict.items()
                ],
                "saved path: {}".format(path),
            )


def is_main_process(func=None):
    """
    check if current process is main process in ddp
    warning: if not in ddp mode, always return True
    :return:
    """
    def _is_main_proc():
        if dist.is_initialized():
            return dist.get_rank() == 0
        else:
            return True
        
    if func is None:
        return _is_main_proc()
    else:
        def warp_func(*args, **kwargs):
            if _is_main_proc():
                return func(*args, **kwargs)
            else:
                return None
            
        return warp_func

def print_args(args):
    
    d = args.__dict__
    for k, v in d.items():
        print(f"{k}: {v}")


def yaml_load(name, base_path="./configs"):
    path = osp.join(base_path, name + "_config.yaml")
    if osp.exists(path):
        f = open(path)
        cont = f.read()
        return yaml.load(cont, Loader=yaml.FullLoader)
    else:
        print("configuration file not exists")
        raise FileNotFoundError(f'file not exists: {path}')


def json_load(name, base_path="./configs"):
    path = osp.join(base_path, name + "_config.json")
    with open(path) as f:
        return json.load(f)


def config_py_load(name, base_path="configs"):
    args = importlib.import_module(f".{name}_config", package=base_path)
    return args.config


class NameSpace:
    def to_dict(self):
        out = {}
        d = self.__dict__
        for k, v in d.items():
            if isinstance(v, NameSpace):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

    def __repr__(self, d=None, nprefix=0):
        repr_str = ""
        if d is None:
            d = self.__dict__
        for k, v in d.items():
            if isinstance(v, NameSpace):
                repr_str += (
                    "  " * nprefix
                    + f"{k}: \n"
                    + f"{self.__repr__(v.__dict__, nprefix + 1)}"
                )
            else:
                repr_str += "  " * nprefix + f"{k}: {v}\n"

        return repr_str


def recursive_search_dict2namespace(d: Dict):
    """
    convert a yaml-like configuration (dict) to namespace-like class

    e.g.
    {'lr': 1e-3, 'path': './datasets/train_wv3.h5'} ->
    NameSpace().lr = 1e-3, NameSpace().path = './datasets/train_wv3.h5'

    Warning: the value in yaml-like configuration should not be another dict
    :param d:
    :return:
    """
    namespace = NameSpace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(namespace, k, recursive_search_dict2namespace(v))
        else:
            setattr(namespace, k, v)

    return namespace


def merge_args_namespace(parser_args: argparse.Namespace, namespace_args: NameSpace):
    """
    merge parser_args and self-made class _NameSpace configurations together for better
    usage.
    return args that support dot its member, like args.optimizer.lr
    :param parser_args:
    :param namespace_args:
    :return:
    """
    # namespace_args.__dict__.update(parser_args.__dict__)
    namespace_d = namespace_args.__dict__
    for k, v in parser_args.__dict__.items():
        if not (k in namespace_d.keys() and v is None):
            setattr(namespace_args, k, v)

    return namespace_args


def generate_id(length: int = 8) -> str:
    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return str(run_gen.random(length))


def find_weight(weight_dir="./weight/", id=None, func=None):
    """
    return weight absolute path referring to id
    Args:
        weight_dir: weight dir that saved weights
        id: weight id
        func: split string function

    Returns: str, absolute path

    """
    assert id is not None, "@id can not be None"
    weight_list = os.listdir(weight_dir)
    if func is None:
        func = lambda x: x.split(".")[0].split("_")[-1]
    for id_s in weight_list:
        only_id = func(id_s)
        if only_id == id:
            return os.path.abspath(os.path.join(weight_dir, id_s))
    print(f"can not find {id}")
    return None


def _delete_unneeded_weight_file(weight_dir="./weight/", id=None):
    """
    delete unneeded weight file referring to id
    Args:
        weight_dir:
        id:

    Returns:

    """
    assert id is not None, "@id can not be None"
    abspath = find_weight(weight_dir, id)
    if abspath is not None:
        assert os.path.exists(abspath)
        os.remove(abspath)
        print(f"delete {os.path.basename(abspath)}")


def print_network_params_macs_fvcore(network, *inputs):
    """
    print out network's parameters and macs by using
    fvcore package
    Args:
        network: nn.Module
        *inputs: input argument

    Returns: None

    """
    analysis = FlopCountAnalysis(network, inputs=inputs)
    print(flop_count_table(analysis))


def clip_dataset_into_small_patches(
    file: h5py.File,
    patch_size: int,
    up_ratio: int,
    ms_channel: int,
    pan_channel: int,
    dataset_keys: Union[list[str], tuple[str]] = ("gt", "ms", "lms", "pan"),
    save_path: str = "./data/clip_data.h5",
):
    """
    clip patches at spatial dim
    Args:
        file: h5py.File of original dataset
        patch_size: ms clipped size
        up_ratio: shape of lms divide shape of ms
        ms_channel:
        pan_channel:
        dataset_keys: similar to [gt, ms, lms, pan]
        save_path: must end with h5

    Returns:

    """
    unfold_fn = lambda x, c, ratio: (
        torch.nn.functional.unfold(
            x, kernel_size=patch_size * ratio, stride=patch_size * ratio
        )
        .transpose(1, 2)
        .reshape(-1, c, patch_size * ratio, patch_size * ratio)
    )

    assert len(dataset_keys) == 4, "length of @dataset_keys should be 4"
    assert save_path.endswith("h5"), "saved file should end with h5 but get {}".format(
        save_path.split(".")[-1]
    )
    gt = unfold_fn(torch.tensor(file[dataset_keys[0]][:]), ms_channel, up_ratio)
    ms = unfold_fn(torch.tensor(file[dataset_keys[1]][:]), ms_channel, 1)
    lms = unfold_fn(torch.tensor(file[dataset_keys[2]][:]), ms_channel, up_ratio)
    pan = unfold_fn(torch.tensor(file[dataset_keys[3]][:]), pan_channel, up_ratio)

    print("clipped datasets shape:")
    print("{:^20}{:^20}{:^20}{:^20}".format(*[k for k in dataset_keys]))
    print(
        "{:^20}{:^20}{:^20}{:^20}".format(
            str(gt.shape), str(ms.shape), str(lms.shape), str(pan.shape)
        )
    )

    base_path = os.path.dirname(save_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"make dir {base_path}")

    save_file = h5py.File(save_path, "w")
    for k, data in zip(dataset_keys, [gt, ms, lms, pan]):
        save_file.create_dataset(name=k, data=data)
        print(f"create data {k}")

    file.close()
    save_file.close()
    print("file closed")
    
def dist_gather_object(obj, n_ranks=1, dest=0, all_gather=False):
    def _iter_tensor_to_rank(rank_obj, dest=0):
        if isinstance(rank_obj, dict):
            for k, v in rank_obj.items():
                if isinstance(v, torch.Tensor):
                    rank_obj[k] = v.to(dest)
                elif isinstance(v, Iterable):
                    rank_obj[k] = _iter_tensor_to_rank(v, dest)
        elif isinstance(rank_obj, (list, tuple)):
            if isinstance(rank_obj, tuple):
                rank_obj = list(rank_obj)
            for i, v in enumerate(rank_obj):
                if isinstance(v, torch.Tensor):
                    rank_obj[i] = v.to(dest)
                elif isinstance(v, Iterable):
                    rank_obj[i] = _iter_tensor_to_rank(v, dest)
        elif isinstance(rank_obj, torch.Tensor):
                rank_obj = rank_obj.to(dest)
    
        return rank_obj
    
    if n_ranks == 1:
        return obj
    elif n_ranks > 1:
        rank_objs = [None] * n_ranks
        if all_gather:
            # all proc to proc dest
            dist.all_gather_object(rank_objs, obj)
            # if is_main_process():
            #     _scattered_objs_lst = [rank_objs] * n_ranks
            # else:
            #     _scattered_objs_lst = [None] * n_ranks
            # received_objs = [None]
            # dist.scatter_object_list(received_objs, _scattered_objs_lst)
            rank_objs = _iter_tensor_to_rank(rank_objs, dest=dest)
        else:
            dist.gather_object(obj, rank_objs if is_main_process() else None, dest)
            if is_main_process():
                rank_objs = _iter_tensor_to_rank(rank_objs, dest)
        return rank_objs
    else:
        raise ValueError("n_ranks should be greater than 0")
    

if __name__ == "__main__":
    # path = "/home/ZiHanCao/datasets/HISI/new_harvard/x8/test_harvard(with_up)x8_rgb.h5"
    # file = h5py.File(path)
    # clip_dataset_into_small_patches(
    #     file,
    #     patch_size=16,
    #     up_ratio=8,
    #     ms_channel=31,
    #     pan_channel=3,
    #     dataset_keys=["GT", "LRHSI", "HSI_up", "RGB"],
    #     save_path="/home/ZiHanCao/datasets/HISI/new_harvard/x8/test_clip_128.h5",
    # )


    vis = torch.randn(1, 3, 256, 256).clip(0, 1)
    ir =  torch.randn(1, 1, 256, 256).clip(0, 1)

    
    model = lambda vis, ir: vis

    with y_pred_model_colored(vis, enable=True) as (y, back_to_rgb):
        pred_y = model(y, ir)
        pred_rgb = back_to_rgb(pred_y)
        
    # assert equal
    print(torch.isclose(pred_rgb, vis))
    
    mean_diff = torch.mean(torch.abs(vis - pred_rgb))
    print("mean difference:", mean_diff.item())