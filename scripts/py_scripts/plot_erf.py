from multiprocessing import heap
import os
import sys
from typing import Sequence
import numpy as np
from timm.utils import AverageMeter
import torch as th
from torch import nn
from torch.nn import functional as F
from kornia.augmentation import CenterCrop
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
import seaborn as sns
from types import SimpleNamespace
from utils import easy_logger

logger = easy_logger(func_name=__name__)

# Set figure parameters
plt.rcParams["font.family"] = "Times New Roman"
large = 24
med = 24
small = 24 
sns_text_size = 4
params = {'axes.titlesize': large,
            'legend.fontsize': med,
            'figure.figsize': (16, 10),
            'axes.labelsize': med,
            'xtick.labelsize': med,
            'ytick.labelsize': med,
            'figure.titlesize': large}
plt.rcParams.update(params)
try:
    plt.style.use('seaborn-whitegrid')
except:
    plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white") 
sns.set_theme(font_scale=sns_text_size)
plt.rcParams['axes.unicode_minus'] = False


class VisERF:
    def __init__(self, 
                 model,
                 data_loader,
                 heat_map_dest='visualized_img/heatmap.png',
                 sample_names: list=['x', 'cond'],
                 key_for_grad: list[str] = ['x'],
                 crop_size: int=128):
        self.model = model
        self.data_loader = data_loader
        self.device = next(model.parameters()).device
        self.heat_map_dest = heat_map_dest
        self.sample_names = sample_names
        self.key_for_grad = key_for_grad
        self.clip_size = crop_size
        self.crop_fn = CenterCrop(self.clip_size)
        os.makedirs(os.path.dirname(heat_map_dest), exist_ok=True)
        logger.info(f'set model to eval mode and device to {self.device}')
        logger.info('heat map will be saved at {}'.format(self.heat_map_dest))
        logger.info('sample names:', sample_names)
        assert self.data_loader.batch_size == 1, 'batch size should be 1 for visualization'
        
    @staticmethod
    def analyze_erf_heatmap(source, dest="heatmap.png", ALGRITHOM=lambda x: np.power(x - 1, 0.2)):
        def _heatmap(data, camp='RdYlGn', figsize=(10, 10), ax=None, save_path=None, cbar=True):
            plt.figure(figsize=figsize, dpi=40)
            ax = sns.heatmap(data,
                            xticklabels=False,
                            yticklabels=False, cmap=camp,
                            center=0, annot=False, ax=ax, cbar=cbar, annot_kws={"size": 24}, fmt='.2f') 
            if cbar: 
                ax.collections[0].set_clim(0,1) 
            plt.savefig(save_path)
            logger.info(f'save heatmap at {save_path}')

        def _analyze_erf(args):
            data = args.source
            logger.info(np.min(data), np.max(data))
            data = args.ALGRITHOM(data + 1)  # the scores differ in magnitude. take the logarithm for better readability
            data = data / np.max(data)  # rescale to [0,1] for the comparability among models
            _heatmap(data, save_path=args.heatmap_save)
            logger.info(f'heatmap saved at {args.heatmap_save}')


        args = SimpleNamespace()
        args.source = source
        args.heatmap_save = dest
        args.ALGRITHOM = ALGRITHOM
        _analyze_erf(args)
        
    # copied from https://github.com/DingXiaoH/RepLKNet-pytorch
    @staticmethod
    def get_input_grad(model,
                       inputs: dict,
                       key_for_grad: str | list[str] = None) -> np.ndarray | list[np.ndarray]: 
        assert key_for_grad is not None, 'key_for_grad should not be None'
        
        outputs = model(**inputs)
        out_size = outputs.size()
        central_point = th.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
        if isinstance(key_for_grad, str):
            grad = th.autograd.grad(central_point, inputs[key_for_grad])
            grad = grad[0]
            grad = th.nn.functional.relu(grad)
            aggregated = grad.sum((0, 1))
            grad_map = aggregated.cpu().numpy()
        elif isinstance(key_for_grad, list):
            grad_map = []
            for key in key_for_grad:
                grad = th.autograd.grad(central_point, inputs[key])
                grad = grad[0]
                grad = th.nn.functional.relu(grad)
                aggregated = grad.sum((0, 1))
                grad_map.append(aggregated.cpu().numpy())
        else:
            raise ValueError('key_for_grad should be a str or a list of str')
        return grad_map
            
    @th.enable_grad()
    def process_erf(self, num_images: int, th_type: th.dtype=th.float32):
        model = self.model
        data_loader = self.data_loader
        # optimizer = th.optim.SGD(model.parameters(), lr=0, weight_decay=0)

        meter = AverageMeter()
        # optimizer.zero_grad()
        H = W = self.clip_size
    
        for idx, samples in enumerate(data_loader):
            samples = samples[:2]  # vi and ir images
            samples_dict = {}
            if isinstance(samples, Sequence):
                for sample_name, sample in zip(self.sample_names, samples):
                    samples_dict[sample_name] = (self.crop_fn(sample)
                                                 .type(th_type)
                                                 .to(self.device, non_blocking=True))
                    samples_dict[sample_name].requires_grad_()
            elif isinstance(samples, th.Tensor):
                samples_dict[self.sample_names[0]] = (self.crop_fn(samples)
                                                      .type(th_type)
                                                      .to(self.device, non_blocking=True))
                samples_dict[self.sample_names[0]].requires_grad_()
                    
            if meter.count == num_images:
                logger.info(f'reach the maximum number of images {num_images}, stop processing')
                break
            
            # optimizer.zero_grad()
            contribution_scores = self.get_input_grad(model, samples_dict, key_for_grad=self.key_for_grad)
            th.cuda.empty_cache()
            
            if isinstance(contribution_scores, np.ndarray):
                if np.isnan(np.sum(contribution_scores)):
                    logger.warning('got NAN, next image')
                    continue
                else:
                    logger.info(f'accumulat{idx}')
                    meter.update(contribution_scores)
            elif isinstance(contribution_scores, list):
                for score in contribution_scores:
                    if np.isnan(np.sum(score)):
                        logger.warning('got NAN, next score')
                        continue
                    else:
                        logger.info(f'accumulat {idx}')
                        meter.update(score)

        grad_avg = meter.avg
        grad_avg = grad_avg.reshape(H, W)
        grad_avg = grad_avg / np.max(grad_avg)
        
        self.analyze_erf_heatmap(grad_avg, dest=self.heat_map_dest)
        
        
os.environ['T_MAX'] = str(128 * 128)
from model.panrwkv_v8_cond_norm import RWKVFusion
class RWKVFusionForERF(RWKVFusion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    def forward(self, x, cond):
        *_, H, W = x.shape
        cat_x_cond = th.cat([x, cond], dim=1)
        x = self.patch_embd(cat_x_cond)

        # U-Net conditioning
        u_cond = cat_x_cond

        if self.if_abs_pos:
            x = x + self.resize_pos_embd(self.abs_pos, inp_size=(H, W))
            
        if self.if_rope:
            x = self.rope(x)

        for encoder, down in zip(self.encoders, self.downs):
            cond1 = F.interpolate(
                u_cond, (H, W), mode="bilinear", align_corners=False
            )
            x = encoder.enc_forward(x, cond1, (H, W))
            x = down(x)
            H = H // 2
            W = W // 2

        cond2 = F.interpolate(
            u_cond, (H, W), mode="bilinear", align_corners=False,
        )
        x = self.middle_blks.enc_forward(x, cond2, (H, W))
        
        return x
    
        
from importlib import import_module

model_name = 'model.panrwkv_v8_cond_norm.RWKVFusion'
model_cfg = {
        "img_channel": 3,
        "condition_channel": 1,
        "out_channel": 3,
        "width": 32,
        "middle_blk_num": 1,
        "enc_blk_nums": [1, 1],
        "dec_blk_nums": [1, 1],
        "chan_upscales": [1, 1],
        "drop_path_rate": 0.0,
        "if_abs_pos": False,
        "if_rope": False,
        "patch_merge": False,
        "upscale": 1,
        "fusion_prior": "max"
}
ckpt_path = "log_file/panRWKV_v8_cond_norm/vis_ir_joint/2024-08-22-01-32-03_panRWKV_r1xf57zs_RWKVFusion_v8_cond_norm/weights/ema_model2.pth"
device = 'cuda:1'
th.cuda.set_device(device)
dataset_cfg = SimpleNamespace(
    dataset = 'msrs',
    path = SimpleNamespace(
        base_dir = "/Data3/cao/ZiHanCao/datasets/MSRS",
    ),
    dataset_mode = 'test',
    reduce_label = True,
)

heat_map_path = f"visualized_img/heap_map_attn.png"


def import_cls(name: str):
    _module = ".".join(name.split('.')[:-1])
    _class = name.split('.')[-1]
    
    return getattr(import_module(_module), _class)

# model
model = RWKVFusionForERF(**model_cfg)
model.load_state_dict(th.load(ckpt_path))
model = model.to(device)

# data
from utils import get_eval_dataset

dataset, _ = get_eval_dataset(dataset_cfg)
data_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

# erf visualization
vis_erf = VisERF(model, data_loader, heat_map_dest=heat_map_path)
vis_erf.process_erf(num_images=20)







