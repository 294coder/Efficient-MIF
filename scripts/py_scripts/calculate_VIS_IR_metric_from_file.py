import os
import numpy as np
import torch
from torchvision.io import read_image

from utils.log_utils import easy_logger
from utils.metric_VIF import AnalysisVISIRAcc


logger = easy_logger(func_name='VISIR_metric')

def read_tensor_img_from_path(path):
    return read_image(path).float()[None].cuda() / 255.

def calculate_VISIR_metric_from_file(fused_path, ir_path, vi_path, metric: AnalysisVISIRAcc):
    fused = read_tensor_img_from_path(fused_path)
    ir = read_tensor_img_from_path(ir_path)
    vi = read_tensor_img_from_path(vi_path)
    
    metric((vi, ir), fused)
    
if __name__ == "__main__":
    from tqdm import tqdm
    
    fused_dir = "/Data3/cao/ZiHanCao/exps/panformer/visualized_img/panRWKV_v8_cond_norm/msrs_v3"
    vi_dir = "/Data3/cao/ZiHanCao/datasets/MSRS/test/raw_png/vi"
    ir_dir = "/Data3/cao/ZiHanCao/datasets/MSRS/test/raw_png/ir"
    
    metric = AnalysisVISIRAcc()
    
    logger.info('start calculating VIS IR metric')
    for fused_name in tqdm(os.listdir(fused_dir), desc="Calculating VIS IR metric"):
        fused_path = os.path.join(fused_dir, fused_name)
        ir_path = os.path.join(ir_dir, fused_name)
        vi_path = os.path.join(vi_dir, fused_name)
        
        calculate_VISIR_metric_from_file(fused_path, ir_path, vi_path, metric)

    logger.info(
        f"VIS IR metric: {metric}"
    )
