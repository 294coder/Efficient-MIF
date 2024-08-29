from collections import OrderedDict
from functools import partial

import os
import h5py
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from scipy.io import savemat

from task_datasets.WV3 import WV3Datasets
from task_datasets.HISR import HISRDatasets
from task_datasets.GF2 import GF2Datasets
from utils import (
    AnalysisPanAcc,
    viz_batch,
    yaml_load,
    res_image,
    unref_for_loop,
    ref_for_loop,
    config_py_load,
    find_key_args_in_log,
    module_load,
)
from model import build_network

device = "cuda:1"
if device != 'cpu':
    torch.cuda.set_device(device)
dataset_type = "gf2"
save_format = "mat"
full_res = False
split_patch = True
patch_size = 64
ergas_ratio = 4
patch_size_list = [
    patch_size // ergas_ratio,
    patch_size // 2,
    patch_size,
    patch_size,
]  # ms, lms, pan
save_mat = False
loop_func = (
    partial(
        ref_for_loop,
        hisi=dataset_type in ["cave", "cave_x8", "harvard", "harvard_x8", "gf5"],
        patch_size_list=patch_size_list,
        ergas_ratio=ergas_ratio,
        residual_exaggerate_ratio=5000,
        sensor=dataset_type,
    )
    if not full_res
    else partial(
        unref_for_loop,
        hisi=dataset_type in ["cave", "cave_x8", "harvard", "harvard_x8", "gf5"],
        patch_size_list=patch_size_list,
        sensor=dataset_type,
    )
)
name = "LEMamba"
subarch = ""
load_from_logs = False
dl_bs = 1
crop_bs = 6


#### print config ####
from loguru import logger

logger.info("=" * 90)
logger.info(f"dataset: {dataset_type}")
logger.info(f"model: {name + (('_' + subarch) if subarch else '')}")
logger.info(f"full_res: {full_res}")
logger.info(f"save_mat: {save_mat}")
if split_patch:
    logger.info(f"patch_size: {patch_size}")
    logger.info(f"patch size list: {patch_size_list}")

print("=" * 90)

# ======================worldview3 checkpoint============
# p = './weight/lformer_R_wv3.pth'  # LFormer

# p = 'weight/panMamba_2ax3fhfp.pth'  # LEMamba
# ========================================================

# ===============GF checkpoint=====================
# p = './weight/lformer_R_gf2.pth'  # LFormer

# p = 'weight/panMamba_313odzav.pth'  # LEMamba
# =================================================

# ================HISI CAVE checkpoint=============
##### cave_x4
# p = './weight/lformer_R_cave_x4.pth'  # LFormer


####### cave_x8

# p = 'weight/2024-04-08-13-33-03_panMamba_3d8t0rg1/panMamba_3d8t0rg1.pth'
##### harvard_x8

# p = './weight/panMamba_1wotinai.pth'  # panMamba
# =================================================



if dataset_type == "wv3":
    if not full_res:
        path = "/Data3/cao/ZiHanCao/datasets/pansharpening/wv3/reduced_examples/test_wv3_multiExm1.h5"
    else:
        # path = '/home/ZiHanCao/datasets/pansharpening/wv3/full_examples/test_wv3_OrigScale_multiExm1.h5'
        path = "/Data3/cao/ZiHanCao/datasets/pansharpening/pansharpening_test/test_wv3_OrigScale_multiExm1.h5"
elif dataset_type == "cave_x4":
    path = "/Data3/cao/ZiHanCao/datasets/HISI/new_cave/test_cave(with_up)x4.h5"
elif dataset_type == "cave_x8":
    path = "/Data2/ZiHanCao/datasets/HISI/new_cave/x8/test_cave(with_up)x8_rgb.h5"
elif dataset_type == "harvard_x4":
    # path = "/Data2/ZiHanCao/datasets/HISI/new_harvard/test_harvard(with_up)x4_rgb.h5"
    path = "/Data2/ShangqiDeng/data/HSI/harvard_x4/test_harvard(with_up)x4_rgb200.h5"
elif dataset_type == "harvard_x8":
    path = "/Data2/ZiHanCao/datasets/HISI/new_harvard/x8/test_harvard(with_up)x8_rgb.h5"
elif dataset_type == "gf5":
    if not full_res:
        path = "/Data2/ZiHanCao/datasets/pansharpening/GF5-GF1/tap23/test_GF5_GF1_23tap_new.h5"
    else:
        path = "/Data2/ZiHanCao/datasets/pansharpening/GF5-GF1/tap23/test_GF5_GF1_OrigScale.h5"
elif dataset_type == "gf2":
    if not full_res:
        path = "/Data2/ZiHanCao/datasets/pansharpening/gf/reduced_examples/test_gf2_multiExm1.h5"
    else:
        path = "/Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_gf2_OrigScale_multiExm1.h5"
elif dataset_type == "qb":
    if not full_res:
        path = "/Data2/ZiHanCao/datasets/pansharpening/qb/reduced_examples/test_qb_multiExm1.h5"
    else:
        path = "/Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_qb_OrigScale_multiExm1.h5"
elif dataset_type == "wv2":
    if not full_res:
        path = "/Data2/ZiHanCao/datasets/pansharpening/wv2/reduced_examples/test_wv2_multiExm1.h5"
    else:
        path = "/Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_wv2_OrigScale_multiExm1.h5"
else:
    raise NotImplementedError("not exists {} dataset".format(dataset_type))

# model = VanillaPANNet(8, 32).to('cuda:0')

if load_from_logs:
    config = find_key_args_in_log(name, subarch, dataset_type, p)
else:
    config = yaml_load(name)
    
full_arch = name + "_" + subarch if subarch != "" else name
model = build_network(full_arch, **(config["network_configs"].get(full_arch, config["network_configs"])))

# -------------------load params-----------------------
model = module_load(p, model, device, strict=True, spec_key='ema_model.shadow_params')
model = model.to(device)
model.eval()
# -----------------------------------------------------

# -------------------get dataset-----------------------
if dataset_type in ["wv3", "qb", "wv2"]:
    d = h5py.File(path)
    ds = WV3Datasets(d, hp=False, full_res=full_res)
elif dataset_type in ["cave_x4", "harvard_x4", "cave_x8", "harvard_x8", "gf5"]:
    d = h5py.File(path)
    d2 = {}
    for k, v in d.items():
        v = v[2:3]
        d2[k] = v
    
    ds = HISRDatasets(d2, full_res=full_res)
elif dataset_type == "gf2":
    d = h5py.File(path)
    ds = GF2Datasets(d, full_res=full_res)
else:
    raise NotImplementedError
dl = data.DataLoader(ds, batch_size=dl_bs)
# -----------------------------------------------------

# -------------------inference-------------------------
all_sr = loop_func(model, dl, device, split_patch=split_patch)
# -----------------------------------------------------

# -------------------save result-----------------------
d = {}
# FIXME: there is an error here, const should be 1023. when sensor is gf
if dataset_type in ["wv3", "qb", "wv2"]:
    const = 2047.0
elif dataset_type in ["gf2"]:
    const = 1023.0
elif dataset_type in [
    "cave_x4",
    "harvard_x4",
    "cave_x8",
    "harvard_x8",
    "roadscene",
    "tno",
    "gf5",
]:
    const = 1.0
else:
    raise NotImplementedError
cat_sr = np.concatenate(all_sr, axis=0).astype("float32")
d["sr"] = np.asarray(cat_sr) * const
try:
    d["gt"] = np.asarray(ds.gt[:]) * const
except:
    print("no gt")
    pass

if save_mat:  # torch.tensor(d['sr'][:, [4,2,0]]),  torch.tensor(d['gt'][:, [4,2,0]])
    _ref_or_not_s = "unref" if full_res else "ref"
    _patch_size_s = f"_p{patch_size}" if split_patch else ""
    if dataset_type not in [
        "cave_x4",
        "harvard_x4",
        "cave_x8",
        "harvard_x8",
        "gf5",
    ]:  # wv3, qb, gf
        d["ms"] = np.asarray(ds.ms[:]) * const
        d["lms"] = np.asarray(ds.lms[:]) * const
        d["pan"] = np.asarray(ds.pan[:]) * const
    else:
        d["ms"] = np.asarray(ds.lr_hsi[:]) * const
        d["lms"] = np.asarray(ds.hsi_up[:]) * const
        d["pan"] = np.asarray(ds.rgb[:]) * const

    if save_format == "mat":
        path = f"./visualized_img/{name}_{subarch}/data_{name}_{subarch}_{dataset_type}_{_ref_or_not_s}{_patch_size_s}.mat"
        os.path.makedirs(os.path.dirname(path), exist_ok=True)
        
        savemat(path, d)
    else:
        path = f"./visualized_img//{name}_{subarch}/data_{name}{subarch}_{dataset_type}_{_ref_or_not_s}{_patch_size_s}.h5"
        save_file = h5py.File(path, "w")
        save_file.create_dataset("sr", data=d["sr"])
        save_file.create_dataset("ms", data=d["ms"])
        save_file.create_dataset("lms", data=d["lms"])
        save_file.create_dataset("pan", data=d["pan"])
        save_file.close()
    print(f"save results in {path}")
# -----------------------------------------------------
