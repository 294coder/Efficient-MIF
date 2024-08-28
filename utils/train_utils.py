from pathlib import Path
import sys

sys.path.append("./")

import zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image
import PIL.Image as Image
from contextlib import contextmanager
from collections import OrderedDict
from typing import Union, TYPE_CHECKING

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import h5py_to_dict, NameSpace, easy_logger

if TYPE_CHECKING:
    from accelerate import Accelerator


def get_eval_dataset(args, logger=None):
    from task_datasets.WV3 import WV3Datasets
    from task_datasets.GF2 import GF2Datasets
    from task_datasets.HISR import HISRDatasets
    from task_datasets.TNO import TNODataset
    from task_datasets.RoadScene import RoadSceneDataset
    from task_datasets.LLVIP import LLVIPDALIPipeLoader
    from task_datasets.MSRS import MSRSDatasets
    from task_datasets.M3FD import M3FDDALIPipeLoader
    from task_datasets.MedHarvard import MedHarvardDataset
    from task_datasets.SICE import SICEDataset

    logger = easy_logger(func_name='get_eval_dataset')

    val_ds, val_dl = None, None

    logger.info(f"use dataset: {args.dataset} on VIS-IR fusion task")
    # 1. vis-ir image fusion (without gt)
    if args.dataset == "roadscene":
        val_ds = RoadSceneDataset(
            args.path.base_dir, "test", no_split=True, get_name=True
        )
    elif args.dataset == "tno":
        val_ds = TNODataset(
            args.path.base_dir, "test", aug_prob=0.0, no_split=True, get_name=True
        )
    elif args.dataset == "msrs":
        val_ds = MSRSDatasets(
            args.path.base_dir,
            mode=args.dataset_mode,  # or 'test'/'detection'
            transform_ratio=0.0,
            get_name=True,
            reduce_label=args.reduce_label,
        )
    elif args.dataset == "llvip":
        val_dl = LLVIPDALIPipeLoader(
            args.path.base_dir,
            "test",
            batch_size=args.val_bs,
            device=args.device,
            shuffle=False,
            with_mask=True,
            get_name=True,
            reduce_label=args.reduce_label,
        )
    elif args.dataset == "m3fd":
        val_dl = M3FDDALIPipeLoader(
            args.path.base_dir,
            "test",
            batch_size=args.val_bs,
            device=args.device,
            shuffle=False,
            with_mask=True,
            get_name=True,
            reduce_label=args.reduce_label,
        )

    elif args.dataset == "med_harvard":
        val_ds = MedHarvardDataset(
            args.path.base_dir,
            mode="test",
            device=args.device,
            data_source="xmu",
            get_name=True,
            task="SPECT-MRI",
        )
    elif args.dataset == "sice":
        val_ds = SICEDataset(
            data_dir=args.path.base_dir,
            mode="test",
            transformer_ratio=0.0,
            only_y=args.only_y,
            get_name=True,
        )

    ## 2. sharpening datasets (with gt)
    elif args.dataset in [
        "wv3",
        "qb",
        "gf2",
        "cave_x4",
        "harvard_x4",
        "cave_x8",
        "harvard_x8",
        "hisi-houston",
        "pavia",
        "chikusei",
        "botswana",
    ]:
        # the dataset has already splitted

        logger.info(f"use dataset: {args.dataset} on pansharpening/HISR task")
        # FIXME: 需要兼顾老代码（只有trian_path和val_path）的情况
        if hasattr(args.path, "val_path"):
            # 旧代码：手动切换数据集路径
            val_path = args.path.val_path
        else:
            _args_path_keys = list(args.path.__dict__.keys())
            for k in _args_path_keys:
                if args.dataset in k:
                    val_path = getattr(args.path, f"{args.dataset}_val_path")
        assert val_path is not None, "val_path should not be None"

        if val_path is not None:
            assert val_path.endswith(".h5"), 'val_path should end with ".h5"'

        h5_val = h5py.File(val_path)

        # 1. parsharpening
        if args.dataset in ["wv3", "qb"]:
            d_val = h5py_to_dict(h5_val)
            val_ds = WV3Datasets(d_val, hp=args.hp, aug_prob=0.0)
        elif args.dataset == "gf2":
            d_val = h5py_to_dict(h5_val)
            val_ds = GF2Datasets(d_val, hp=args.hp, aug_prob=0.0)

        # 2. hyperspectral image fusion
        elif (
            args.dataset[:4] == "cave"
            or args.dataset[:7] == "harvard"
            or args.dataset[:8] == "chikusei"
            or args.dataset[:5] == "pavia"
            or args.dataset[:8] == "botswana"
        ):
            keys = ["LRHSI", "HSI_up", "RGB", "GT"]
            if args.dataset.split("-")[-1] == "houston":
                from einops import rearrange

                # to avoid unpicklable error
                def permute_fn(x):
                    return rearrange(x, "b h w c -> b c h w")

                dataset_fn = permute_fn
            else:
                dataset_fn = None

            d_val = h5py_to_dict(h5_val, keys)
            val_ds = HISRDatasets(
                d_val, aug_prob=args.aug_probs[1], dataset_fn=dataset_fn
            )
            # del h5_train, h5_val
    else:
        raise NotImplementedError(f"not support dataset {args.dataset}")

    return val_ds, val_dl


def get_fusion_dataset(
    args: NameSpace, accelerator: "Accelerator", device: Union[str, torch.device]
):
    logger = easy_logger()
    
    train_ds, val_ds, train_dl, val_dl = None, None, None, None

    if args.dataset in [
        "flir",
        "tno",
        "roadscene_tno_joint",
        "vis_ir_joint",
        "msrs",
        "llvip",
        "med_harvard",
        "m3fd",
        "sice",
        "mefb",
    ]:
        args.task = "fusion"
        args.has_gt = False
        args.path.base_dir = getattr(args.path, f"{args.dataset}_base_dir")
        if args.dataset == "roadscene":
            from task_datasets.RoadScene import RoadSceneDataset

            train_ds = RoadSceneDataset(args.path.base_dir, "train")
            val_ds = RoadSceneDataset(args.path.base_dir, "test")
        elif args.dataset in ["tno", "roadscene_tno_joint"]:
            from task_datasets.TNO import TNODataset

            train_ds = TNODataset(
                args.path.base_dir,
                "train",
                aug_prob=args.aug_probs[0],
                duplicate_vis_channel=True,
            )
            val_ds = TNODataset(
                args.path.base_dir,
                "test",
                aug_prob=args.aug_probs[1],
                no_split=True,
                duplicate_vis_channel=True,
            )
        elif args.dataset == "msrs":
            from task_datasets.MSRS import MSRSDatasets

            train_ds = MSRSDatasets(
                args.path.base_dir,
                "train",
                transform_ratio=args.aug_probs[0],
                output_size=args.fusion_crop_size,
                n_proc_load=1,
                reduce_label=args.datasets_cfg.reduce_label,
                only_y_component=False,  # args.only_y
            )
            val_ds = MSRSDatasets(
                args.path.base_dir,
                "test",
                transform_ratio=args.aug_probs[1],
                output_size=args.fusion_crop_size,
                n_proc_load=1,
                reduce_label=args.datasets_cfg.reduce_label,
                only_y_component=False,  # args.only_y
            )
        elif args.dataset == "llvip":
            from task_datasets.LLVIP import LLVIPDALIPipeLoader

            # We use DALI pipeline to accelerate the data loading process
            train_dl = LLVIPDALIPipeLoader(
                args.path.base_dir,
                "train",
                batch_size=args.train_bs,
                output_size=args.fusion_crop_size,
                device=accelerator.device,
                num_shards=accelerator.num_processes,
                shard_id=accelerator.process_index,
                shuffle=True,
                reduce_label=args.datasets_cfg.reduce_label,
                only_y_component=False,  # args.only_y
            )
            val_dl = LLVIPDALIPipeLoader(
                args.path.base_dir,
                "test",
                batch_size=args.val_bs,
                device=accelerator.device,
                fast_eval_n_samples=args.fast_eval_n_samples,
                num_shards=accelerator.num_processes,
                shard_id=accelerator.process_index,
                shuffle=True,
                reduce_label=args.datasets_cfg.reduce_label,
                only_y_component=False,  # args.only_y
            )
        elif args.dataset == "m3fd":
            from task_datasets import M3FDDALIPipeLoader

            train_dl = M3FDDALIPipeLoader(
                args.path.base_dir,
                "train",
                batch_size=args.train_bs,
                output_size=args.fusion_crop_size,
                device=accelerator.device,
                num_shards=accelerator.num_processes,
                shard_id=accelerator.process_index,
                shuffle=True,
                reduce_label=args.datasets_cfg.reduce_label,
                only_y_component=False,  # args.only_y
            )
            val_dl = M3FDDALIPipeLoader(
                args.path.base_dir,
                "test",
                batch_size=args.val_bs,
                device=accelerator.device,
                fast_eval_n_samples=args.fast_eval_n_samples,
                num_shards=accelerator.num_processes,
                shard_id=accelerator.process_index,
                shuffle=True,
                reduce_label=args.datasets_cfg.reduce_label,
                only_y_component=False,  # args.only_y
            )

        elif args.dataset == "vis_ir_joint":
            from task_datasets import VISIRJointGenericLoader

            train_dl = VISIRJointGenericLoader(
                vars(args.path.base_dir),
                mode="train",
                batch_size=args.train_bs,
                output_size=args.fusion_crop_size,
                device=accelerator.device,
                shuffle_in_dataset=True,
                num_shards=accelerator.num_processes,
                shard_id=accelerator.process_index,
                reduce_label=args.datasets_cfg.reduce_label,
                only_y_component=False,
            )
            val_dl = VISIRJointGenericLoader(
                ## only test msrs and roadscene_tno_joint dataset
                {'msrs': args.path.base_dir['msrs'],
                 'roadscene_tno_joint': args.path.base_dir['roadscene_tno_joint']},
                mode="test",
                output_size=224,  # enforce the different size images to be the same size
                batch_size=args.val_bs,
                device=accelerator.device,
                shuffle_in_dataset=False,
                fast_eval_n_samples=30,
                num_shards=accelerator.num_processes,
                shard_id=accelerator.process_index,
                reduce_label=args.datasets_cfg.reduce_label,
                only_y_component=False,
            )

        elif args.dataset == "med_harvard":
            from task_datasets.MedHarvard import MedHarvardDataset

            if getattr(args, "datasets_cfg", None):
                task = args.datasets_cfg.med_harvard.task
            else:
                task = None

            train_ds = MedHarvardDataset(
                args.path.base_dir,
                mode="train",
                device=device,
                data_source="xmu",
                transform_ratio=args.aug_probs[0],
                task=task,
            )
            val_ds = MedHarvardDataset(
                args.path.base_dir,
                mode="test",
                device=device,
                data_source="xmu",
                task=task,
            )

            assert args.num_worker == 0, "num_worker should be 0 for MedHarvard dataset"

        elif args.dataset == "sice":
            from task_datasets.SICE import SICEDataset

            train_ds = SICEDataset(
                data_dir=args.path.base_dir,
                mode="train",
                transformer_ratio=args.aug_probs[0],
                output_size=args.fusion_crop_size,
                only_y=args.only_y,
            )
            val_ds = SICEDataset(
                data_dir=args.path.base_dir,
                mode="train",
                transformer_ratio=args.aug_probs[0],
                output_size=args.fusion_crop_size,
                only_y=args.only_y,
            )
            args.has_gt = True

        else:
            raise NotImplementedError(f"not support dataset {args.dataset}")

    elif args.dataset in [
        "wv3",
        "qb",
        "gf2",
        "cave_x4",
        "harvard_x4",
        "cave_x8",
        "harvard_x8",
        "hisi-houston",
    ]:
        args.task = "sharpening"

        # the dataset has already splitted
        # FIXME: 需要兼顾老代码（只有trian_path和val_path）的情况
        if hasattr(args.path, "train_path") and hasattr(args.path, "val_path"):
            # 旧代码：手动切换数据集路径
            train_path = args.path.train_path
            val_path = args.path.val_path
        else:
            _args_path_keys = list(args.path.__dict__.keys())
            for k in _args_path_keys:
                if args.dataset in k:
                    train_path = getattr(args.path, f"{args.dataset}_train_path")
                    val_path = getattr(args.path, f"{args.dataset}_val_path")
        assert (
            train_path is not None and val_path is not None
        ), "train_path and val_path should not be None"

        h5_train, h5_val = (
            h5py.File(train_path),
            h5py.File(val_path),
        )

        if args.dataset in ["wv3", "qb"]:
            from task_datasets.WV3 import WV3Datasets, make_datasets

            d_train, d_val = h5py_to_dict(h5_train), h5py_to_dict(h5_val)
            train_ds, val_ds = (
                WV3Datasets(d_train, aug_prob=args.aug_probs[0]),
                WV3Datasets(d_val, aug_prob=args.aug_probs[1]),
            )
        elif args.dataset == "gf2":
            from task_datasets.GF2 import GF2Datasets

            d_train, d_val = h5py_to_dict(h5_train), h5py_to_dict(h5_val)
            train_ds, val_ds = (
                GF2Datasets(d_train, aug_prob=args.aug_probs[0]),
                GF2Datasets(d_val, aug_prob=args.aug_probs[1]),
            )
        elif args.dataset[:4] == "cave" or args.dataset[:7] == "harvard":
            from task_datasets.HISR import HISRDatasets

            keys = ["LRHSI", "HSI_up", "RGB", "GT"]
            if args.dataset.split("-")[-1] == "houston":
                from einops import rearrange

                def permute_fn(x):
                    return rearrange(x, "b h w c -> b c h w")

                dataset_fn = permute_fn
            else:
                dataset_fn = None

            d_train, d_val = (
                h5py_to_dict(h5_train, keys),
                h5py_to_dict(h5_val, keys),
            )
            train_ds = HISRDatasets(
                d_train, aug_prob=args.aug_probs[0], dataset_fn=dataset_fn
            )
            val_ds = HISRDatasets(
                d_val, aug_prob=args.aug_probs[1], dataset_fn=dataset_fn
            )
            # del h5_train, h5_val
        else:
            raise NotImplementedError(f"not support dataset {args.dataset}")

    train_sampler, val_sampler = None, None

    if train_dl is None:
        train_dl = DataLoader(
            train_ds,
            args.train_bs,
            num_workers=args.num_worker,
            sampler=train_sampler,
            prefetch_factor=8 if args.num_worker > 0 else None,
            pin_memory=False,
            shuffle=args.shuffle if not args.ddp else None,
            drop_last=True if args.shuffle else False,
        )
    if val_dl is None:
        val_dl = DataLoader(
            val_ds,
            args.val_bs,  # assert bs is 1, when using PatchMergeModule
            num_workers=0,
            sampler=val_sampler,
            pin_memory=False,
            shuffle=args.shuffle if not args.ddp else None,
            drop_last=False,
        )

    return train_ds, train_dl, val_ds, val_dl


def set_ema_model_params_with_keys(ema_model_params: "dict[str, list[torch.Tensor] | int | float]", 
                                   keys: "list[str]",
                                   keys_set: list[str]=['shadow_params']):
    """set ema model parameters with keys

    Args:
        ema_model_params (dict[str, list[torch.Tensor] | int | float]): ema model parameters
        keys (list[str]): keys

    Returns:
        dict: ema model parameters with keys
    """
    logger = easy_logger()
    
    if not isinstance(keys, list):
        keys = list(keys)
    
    ema_model_params_with_keys = OrderedDict()
    for k in ema_model_params.keys():
        if k in keys_set and k in ema_model_params:
            logger.info(f'set ema_model {k} params with keys')
            params = ema_model_params[k]
            assert params is not None
            assert len(params) == len(keys), "ema_model_params and keys should have the same length"
            
            _params = OrderedDict()
            for mk, p in zip(keys, params):
                _params[mk] = p
                
            ema_model_params_with_keys[k] = _params
        elif k not in keys_set and k in ema_model_params:
            ema_model_params_with_keys[k] = ema_model_params[k]
            
    return ema_model_params_with_keys


def run_once(abled=True):
    def _inner(func):
        def _wrapper(*args, **kwargs):
            nonlocal abled
            if not abled:
                return None
            else:
                outs = func(*args, **kwargs)
                abled = False
                return outs

        return _wrapper

    return _inner


def sanity_check(func: callable):
    @run_once()
    def _inner(*args, **kwargs):
        return func(*args, **kwargs)

    return _inner


@contextmanager
def save_imgs_in_zip(
    zipfile_name: str, mode="w", verbose: bool = False, save_file_ext: str = "jpeg"
):
    """save images to a zip file

    Args:
        zipfile_name (str): zip filename
        mode (str, optional): mode to write in. Defaults to "w".
        verbose (bool, optional): print out. Defaults to False.
        save_file_ext (str, optional): image extension in the zip file. Defaults to "jpeg".

    Yields:
        callable: a function to save image

    Examples::

        with save_imgs_in_zip('zip_file.zip') as add_image:
            img, img_name = get_img()
            add_image(img, img_name)

    :ref: `add_image`

    """
    logger = easy_logger()
    
    # save_file_ext = save_file_ext.upper()
    zf = zipfile.ZipFile(
        zipfile_name, mode=mode, compression=zipfile.ZIP_DEFLATED, compresslevel=9
    )
    bytes_io = BytesIO()
    # jpg compression
    _jpg_quality = 100  # 95 if save_file_ext in ["jpeg", "jpg", "JPG", "JPEG"] else 100

    try:

        logger.info(f"zip file will be saved at {zipfile_name}")

        def to_bytes(image_data, image_name):
            batched_image_bytes = []

            if image_data.ndim == 4:  # batched rgb images
                assert isinstance(image_name, list), "image_name should be a list"
                assert image_data.shape[0] == len(
                    image_name
                ), "image_name should have the same length as image_data"

                for img in image_data:  # [b, h, w, c]
                    Image.fromarray(img).save(
                        bytes_io, format=save_file_ext, quality=_jpg_quality
                    )
                    batched_image_bytes.append(bytes_io.getvalue())
            elif image_data.ndim == 3:
                if image_data.shape[-1] == 1:  # gray image  # [h, w, 1]
                    Image.fromarray(image_data[..., 0]).save(
                        bytes_io, format=save_file_ext, quality=_jpg_quality
                    )
                    image_data = bytes_io.getvalue()
                elif image_data.shape[-1] == 3:
                    Image.fromarray(image_data).save(
                        bytes_io, format=save_file_ext, quality=_jpg_quality
                    )
                    image_data = bytes_io.getvalue()
                else:
                    raise ValueError(
                        f"image_data shape {image_data.shape} not supported"
                    )
            elif image_data.ndim == 2:  # gray image  # [h, w]
                Image.fromarray(image_data).save(
                    bytes_io, format=save_file_ext, quality=_jpg_quality
                )
                image_data = bytes_io.getvalue()

            return image_data, batched_image_bytes

        def add_image(
            image_data: "Image.Image | np.ndarray | torch.Tensor | bytes",
            image_name: "Union[str, list[str]]",
        ):
            """add image to the zipfile

            Args:
                image_data (Image.Image | np.ndarray | torch.Tensor | bytes): can be Image.Image, np.ndarray, torch.Tensor, bytes,
                                                    shape should be [b, h, w, c], [h, w, c], [h, w, 1]
                image_name (str | list[str]): saved image names
            """

            # to bytes
            batched_image_bytes = None
            if isinstance(image_data, Image.Image):
                image_data.save(bytes_io, format=save_file_ext)
                bytes = bytes_io.getvalue()
            elif isinstance(image_data, np.ndarray):
                bytes, batched_image_bytes = to_bytes(image_data, image_name)
            elif isinstance(image_data, torch.Tensor):
                image_data = image_data.detach().cpu().numpy()
                bytes, batched_image_bytes = to_bytes(image_data, image_name)
            else:
                raise ValueError(f"image_data type {type(image_data)} not supported")

            # saving to zip file
            if batched_image_bytes is not None:
                for i, img_bytes in enumerate(batched_image_bytes):
                    zf.writestr(image_name[i], img_bytes)
            else:
                zf.writestr(image_name, bytes)

            if verbose:
                logger.info(f"add image {image_name} to zip file")

            bytes_io.seek(0)
            bytes_io.truncate()

        yield add_image

    except Exception as e:
        if verbose:
            logger.error(e, raise_error=True)
            raise e
    finally:
        if verbose:
            logger.info(f"zip file saved at {zipfile_name}, zipfile close")
        zf.close()
        bytes_io.close()

