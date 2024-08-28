from pathlib import Path
import torch
from torch.utils.data import DataLoader
import sys
import importlib
import accelerate
from accelerate.utils import set_seed, gather_object, gather
from accelerate import Accelerator
from PIL.Image import fromarray as PIL_from_array
from rich.console import Console
from argparse import ArgumentParser
from omegaconf import OmegaConf

from model import build_network
from utils import AnalysisVISIRAcc, EasyProgress, easy_logger
from utils.train_utils import get_eval_dataset
from utils.load_params import module_load
from utils.misc import WindowBasedPadder

# logger
from utils import LoguruLogger, y_pred_model_colored

logger = LoguruLogger.logger()

def ascii_tensor_to_string(ascii_tensor):
    # batched tensor of ascii code
    
    ascii_array = ascii_tensor.detach().cpu().numpy()
    
    string_s = []
    for arr in ascii_array:
        characters = [chr(code) for code in arr]
        string_s.append(''.join(characters))
        
    return string_s

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, default=None, help='config file path')
    parser.add_argument('-m', '--model-class', type=str, default=None, help='model class name')
    parser.add_argument('--arch', type=str, default=None)
    parser.add_argument('--sub-arch', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='tno')
    parser.add_argument('--val-bs', type=int, default=2)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--load-spec-key', type=str, default=None)
    parser.add_argument('--reduce-label', action='store_true', default=False)
    parser.add_argument('--dataset-mode', type=str, default='test', choices=['test', 'detection'])
    parser.add_argument('--save-path', type=str, default='visualized_img/')
    parser.add_argument('--extra-save-name', type=str, default='')
    parser.add_argument('--only-y', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--analysis-fused', action='store_true', default=False, help='analysis fused image')
    
    args = parser.parse_args()
    
    # to omegaconf
    conf = OmegaConf.create(args.__dict__)
    
    # load network config
    # old model loadding
    if args.model_class is None:
        conf.full_arch = args.arch + "_" + args.sub_arch if args.sub_arch is not None else args.arch
        conf.network_configs_path = Path('configs') / f'{conf.arch}_config.yaml'
        yaml_cfg = OmegaConf.load(conf.network_configs_path)
        conf.network_configs = getattr(yaml_cfg['network_configs'], conf.full_arch, yaml_cfg['network_configs'])
    # new model loadding
    else:
        assert args.config_file is not None, 'config file should be provided'
        yaml_cfg = OmegaConf.load(args.config_file)
        model_init_kwargs = getattr(yaml_cfg.network_configs, args.model_class, yaml_cfg.network_configs)
        conf.full_arch = conf.model_class.replace('.', '_')
        conf.network_configs = model_init_kwargs
    
    # dataset and save path config
    conf.dataset = conf.dataset.lower()
    conf.path = yaml_cfg.path
    conf.path.base_dir = getattr(conf.path, f'{conf.dataset}_base_dir')
    conf.save_path = Path(args.save_path) / conf.full_arch / (conf.dataset + (f'_{args.extra_save_name}' if args.extra_save_name else ''))
    
    return conf


def inference_main(args):
    accelerator = Accelerator(mixed_precision='no')
    set_seed(2024)
    
    # logger
    logger = easy_logger()
    
    # device
    args.device = str(accelerator.device)
        
    # multiprocessing config
    device = accelerator.device
    n_process = accelerator.num_processes
    is_main_process = accelerator.is_main_process
    use_ddp = n_process > 1
    # logger = logger.bind(proc=accelerator.process_index)
    _n_saved_imgs = 0
    
    only_rank_zero_print = logger.info if is_main_process else lambda *args: None
    
    # dataset config
    val_ds, val_dl = get_eval_dataset(args, logger)
    if val_ds is not None:
        val_dl = DataLoader(val_ds, batch_size=args.val_bs, shuffle=False)
    
    # model config
    if args.model_class is None:
        network = build_network(args.full_arch, **args.network_configs)
    else:
        logger.info(f'loading model from class: {args.model_class} with config file: {args.config_file}')
        _module_name, _class_name = args.model_class.split('.')
        _module = importlib.import_module(_module_name, package='model')
        network = getattr(_module, _class_name)(**dict(args.network_configs, ))
        
    network = module_load(args.model_path, network, device=device, spec_key=args.load_spec_key)
    
    analysor = AnalysisVISIRAcc()
    padder = WindowBasedPadder(32)
    
    # prepare network, dataloader, and image saved path
    network, val_dl = accelerator.prepare(network, val_dl)
    network.eval()
    
    if is_main_process:
        save_path = args.save_path
        save_path.mkdir(exist_ok=True, parents=True)
        only_rank_zero_print(f'Ready to save images at: {save_path}')
    
    # for-loop inference
    tbar, task_id = EasyProgress.easy_progress(['inference'], [len(val_dl)],
                                               is_main_process=is_main_process,
                                               start_tbar=True, debug=args.debug,
                                               tbar_kwargs={'console': logger._console})
    only_rank_zero_print('start inference...')
    for i, data in enumerate(val_dl):
        if len(data) == 5:
            vis, ir, mask, gt, file_name = data
            split_context = accelerator.split_between_processes(dict(vis=vis, ir=ir, mask=mask, gt=gt, file_name=file_name))
        elif len(data) == 4:
            vis, ir, gt, file_name = data
            split_context = accelerator.split_between_processes(dict(vis=vis, ir=ir, gt=gt, file_name=file_name))
        else:
            raise ValueError('data should have 4 or 5 elements.')
        
        with torch.no_grad() and torch.inference_mode() and split_context as split_d:
            vis, ir, gt, file_name = split_d['vis'], split_d['ir'], split_d['gt'], split_d['file_name']
            if 'mask' in split_d:
                mask = split_d['mask']
            else:
                mask = None
            
            ir = padder(ir)
            vis = padder(vis, no_check_pad=True)
            with y_pred_model_colored(vis, enable=args.only_y) as (vis_y, back_to_rgb):
                fused = network(vis_y, ir, mask, mode='fusion_eval')#, ret_seg_map=False)
                fused = back_to_rgb(fused)
            fused = padder.inverse(fused)
            if args.analysis_fused:
                analysor(gt, fused)
    
            # save figs
            if use_ddp:
                gathered_fused = gather(fused).reshape(-1, *fused.size()[1:])  # [np * b, c, h, w]
                if isinstance(file_name, (list, tuple)):
                    _gathered_file_names = gather_object(file_name)
                    gathered_file_names = []
                    for batched_file_names in _gathered_file_names:
                        for name in batched_file_names:
                            gathered_file_names.append(name)
                elif isinstance(file_name, torch.Tensor):
                    gathered_file_names = gather(file_name)
                    gathered_file_names = ascii_tensor_to_string(gathered_file_names)
            else:
                gathered_fused = fused
                if isinstance(file_name, (list, tuple)):
                    gathered_file_names = file_name
                else:
                    gathered_file_names = ascii_tensor_to_string(file_name)
            
            assert gathered_fused.size(0) == len(gathered_file_names), 'gathered_fused and gathered_file_names should have the same length.'
                
            if is_main_process:
                for idx, fused in enumerate(gathered_fused):
                    save_name = save_path / gathered_file_names[idx]
                    fused = fused.permute(1, 2, 0).cpu().numpy()  # [h, w, c]
                    fused = (fused * 255).astype('uint8')
                    pil_img = PIL_from_array(fused)
                    if args.dataset.lower() in ['tno', 'roadscene']:
                        pil_img = pil_img.convert("L")
                    logger.info(f'saving figs {gathered_file_names[idx]} ...')
                    pil_img.save(save_name)
                _n_saved_imgs += gathered_fused.size(0)
                
            # advance progress bar
            if not args.debug:
                tbar.update(task_id, completed=i+1, total=len(val_dl), 
                            visible=True if i+1 < len(val_dl) else False,
                            description='Inference...')
                        
    
    # print results
    if use_ddp:
        analysors = gather_object(analysor)
        analysor.ave_result_with_other_analysors(analysors, ave_to_self=True)
        only_rank_zero_print(analysor.result_str())
    else:
        logger.info(analysor.result_str())
        
    only_rank_zero_print('Inference Done.')
    

if __name__ == '__main__':
    args = get_args()
    
    try:
        inference_main(args)
    except Exception as e:
        EasyProgress.close_all_tasks()
        logger.exception(e)
        raise e
    
    
    
    
        
            
            
    
    
    
    