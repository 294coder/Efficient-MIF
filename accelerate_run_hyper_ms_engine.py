import os
from functools import partial
import math
from typing import Callable, Union

import torch
import accelerate
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from accelerate.utils import DistributedType
import warnings
# filter off all warnings
warnings.filterwarnings("ignore", module="torch")

from utils import (
    AnalysisPanAcc,
    AnalysisVISIRAcc,
    dict_to_str,
    prefixed_dict_key,
    res_image,
    step_loss_backward,
    accum_loss_dict,
    ave_ep_loss,
    loss_dict2str,
    ave_multi_rank_dict,
    NonAnalysis,
    dist_gather_object,
    get_precision,
    module_load,
)

from schedulefree import AdamWScheduleFree
from utils.log_utils import TensorboardLogger
from typing_extensions import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from accelerate.optimizer import AcceleratedOptimizer
    from model.base_model import BaseModel
    

def _optimizer_train(optimizer: "AcceleratedOptimizer"):
    if isinstance(optimizer.optimizer, AdamWScheduleFree):
        optimizer.train()
    return optimizer

def _optimizer_eval(optimizer: "AcceleratedOptimizer"):
    if isinstance(optimizer.optimizer, AdamWScheduleFree):
        optimizer.eval()
    return optimizer

@torch.no_grad()
def val(
        accelerator: accelerate.Accelerator,
        network: "BaseModel",
        val_dl: "DataLoader",
        criterion: callable,
        logger: Union["TensorboardLogger"],
        ep: int = None,
        optim_val_loss: float = None,
        args=None,
):
    val_loss = 0.0
    val_loss_dict = {}
    
    # get analysor for validate metrics
    if args.log_metrics:
        if args.dataset in ['flir', 'tno']: analysis = AnalysisVISIRAcc()
        else: analysis = AnalysisPanAcc(args.ergas_ratio)
    else:
        analysis = NonAnalysis()
    
    dtype = get_precision(accelerator.mixed_precision)
    logger.print('=======================================EVAL STAGE=================================================')
    for i, (pan, ms, lms, gt) in tqdm(enumerate(val_dl, 1), disable=not accelerator.is_main_process, 
                                      total=len(val_dl), leave=False, dynamic_ncols=True):
        
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            pan = pan.to(accelerator.device, dtype=dtype)
            ms = ms.to(accelerator.device, dtype=dtype)
            lms = lms.to(accelerator.device, dtype=dtype)
            gt = gt.to(accelerator.device, dtype=dtype)

        sr = network(ms, lms, pan, mode="eval")

        loss_out = criterion(sr, gt)
        # if loss is hybrid, will return tensor loss and a dict
        if isinstance(loss_out, tuple):
            batched_loss, loss_d = loss_out
        else:
            batched_loss = loss_out
            loss_d = {'val_main_loss': loss_out}

        analysis(gt, sr)
        val_loss += batched_loss
        val_loss_dict = accum_loss_dict(val_loss_dict, loss_d)
    
    logger.print(analysis.result_str(), dist=True, proc_id=accelerator.process_index)  # log in every process

    val_loss /= i
    val_loss_dict = ave_ep_loss(val_loss_dict, i)
    
    if args.ddp:
        if args.log_metrics:  # gather from all procs to proc 0
            _gathered_analysis = dist_gather_object(analysis, n_ranks=accelerator.num_processes)
        gathered_val_dict = dist_gather_object(val_loss_dict, n_ranks=accelerator.num_processes)
        val_loss_dict = ave_multi_rank_dict(gathered_val_dict)

    # gather metrics and log image
    acc_ave = analysis.acc_ave
    if accelerator.is_main_process:
        if args.ddp and args.log_metrics:
            n = 0
            acc = _gathered_analysis[0].empty_acc
            for analysis in _gathered_analysis:
                for k, v in analysis.acc_ave.items():
                    acc[k] += v * analysis._call_n
                n += analysis._call_n
            for k, v in acc.items():
                acc[k] = v / n
            acc_ave = acc
        if logger is not None:
            # log validate curves
            if args.log_metrics:
                logger.log_curves(prefixed_dict_key(acc_ave, "val"), ep)
            logger.log_curve(val_loss, "val_loss", ep)
            for k, v in val_loss_dict.items():
                logger.log_curve(v, f'val_{k}', ep)

            # log validate image(last batch)
            if gt.shape[0] > 8:
                func = lambda x: x[:8]
                gt, lms, pan, sr = list(map(func, [gt, lms, pan, sr]))
            residual_image = res_image(gt, sr, exaggerate_ratio=100)  # [b, 1, h, w]
            logger.log_images([lms, pan, sr, residual_image], nrow=4, names=["lms", "pan", "sr", "res"], epoch=ep, ds_name=args.dataset)

    # print eval info
    logger.print('\n\nsummary of evaluation:')
    if accelerator.is_main_process:
        logger.print(loss_dict2str(val_loss_dict))
        logger.print(f"\n{dict_to_str(acc_ave)}" if args.log_metrics else "")
    logger.print('==================================================================================================')
    return acc_ave, val_loss  # only rank 0 is reduced and other ranks are original data

def train(
        accelerator: accelerate.Accelerator,
        model: "BaseModel",
        optimizer,
        criterion,
        warm_up_epochs,
        lr_scheduler,
        train_dl: "DataLoader",
        val_dl: "DataLoader",
        epochs: int,
        eval_every_epochs: int,
        save_path: str,
        check_save_fn: Callable=None,
        logger: Union["TensorboardLogger"] = None,
        resume_epochs: int = 1,
        ddp=False,
        max_norm=None,
        grad_accum_ep=None,
        args=None,
):
    """
    train and val script
    :param network: Designed network, type: nn.Module
    :param optim: optimizer
    :param criterion: loss function, type: Callable
    :param warm_up_epochs: int
    :param lr_scheduler: scheduler
    :param train_dl: dataloader used in training
    :param val_dl: dataloader used in validate
    :param epochs: overall epochs
    :param eval_every_epochs: validate epochs
    :param save_path: model params and other params' saved path, type: str
    :param logger: Tensorboard logger or Wandb logger
    :param resume_epochs: when retraining from last break, you should pass the arg, type: int
    :param ddp: distribution training, type: bool
    :param fp16: float point 16, type: bool
    :param max_norm: max normalize value, used in clip gradient, type: float
    :param args: other args, see more in main.p y
    :return:
    """    
    save_checker = lambda *check_args: check_save_fn(check_args[0]) if check_save_fn is not None else \
                    lambda val_acc_dict, val_loss, optim_val_loss: val_loss < optim_val_loss
    
    # if warm_up_epochs > 0:
    #     warm_up_scheduler = LinearWarmupScheduler(optim, 0, args.optimizer.lr, warm_up_epochs)
    
    dtype = get_precision(accelerator.mixed_precision)
    
     # load pretrain model
    if args.pretrain_model_path is not None:
        # e.g., panMamba.pth
        assert args.pretrain_model_path.endswith('.pth') or args.pretrain_model_path.endswith('.safetensors')
        model = module_load(args.pretrain_model_path, model, device=accelerator.device, strict=args.non_load_strict)
    
    # Prepare everything with accelerator
    model, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, lr_scheduler
    )
    
    # load state
    if args.resume_path is not None:
        # e.g., panMamba/**/ep_80
        accelerator.load_state(input_dir=args.resume_path)
    
    # FIXME: ZERO3 does not support!
    ema_net = ExponentialMovingAverage(parameters=[p for p in model.parameters() if p.requires_grad],
                                       decay=args.ema_decay)
    ema_net.to(next(model.parameters()).device)

    # Figure out how many steps we should save the Acclerator states
    # checkpointing_steps = args.checkpointing_steps
    # if checkpointing_steps is not None and checkpointing_steps.isdigit():
    #     checkpointing_steps = int(checkpointing_steps)
    
    world_size = args.world_size if ddp else None
    optim_val_loss = math.inf
    fp_scaler = None  # accelerator.scaler if accelerator.mixed_precision != 'fp32' else None
    
    logger.print(f">>> start training!")
    logger.print(f">>> Num Iterations per Epoch= {len(train_dl)}")
    logger.print(f">>> Num Epochs = {args.num_train_epochs}")
    logger.print(f">>> Gradient Accumulation steps = {args.grad_accum_steps}")
    
    optimizer = _optimizer_train(optimizer)
    for ep in range(resume_epochs, epochs + 1):
        # if args.ddp:
        #     train_dl.sampler.set_epoch(ep)
        #     val_dl.sampler.set_epoch(ep)
        
        ep_loss = 0.0
        ep_loss_dict = {}
        i = 0
        # model training
        tbar = tqdm(enumerate(train_dl, 1), disable=not accelerator.is_main_process,
                    total=len(train_dl), leave=False, dynamic_ncols=True,)
        for i, (pan, ms, lms, gt) in tbar:
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                pan = pan.to(accelerator.device, dtype=dtype)
                ms = ms.to(accelerator.device, dtype=dtype)
                lms = lms.to(accelerator.device, dtype=dtype)
                gt = gt.to(accelerator.device, dtype=dtype)
            
            with accelerator.autocast() and accelerator.accumulate(model):
                sr, loss_out = model(ms, lms, pan, gt, criterion, mode="train")
            
            # if loss is hybrid, will return tensor loss and a dict
            if isinstance(loss_out, tuple):
                loss, loss_d = loss_out
            else: loss = loss_out
            
            # if accelerator.sync_gradients:
            #     completed_steps += 1
                    
            ep_loss += loss
            tbar.set_description(f"loss: {loss:.4f}")
            
            if torch.isnan(loss).any():
                raise ValueError(f">>> PROCESS {accelerator.process_index}: loss is nan")
            
            ep_loss_dict = accum_loss_dict(ep_loss_dict, loss_d)

            # update parameters
            step_loss_backward_partial = partial(
                step_loss_backward,
                optim=optimizer,
                network=model,
                max_norm=max_norm,
                loss=loss,
                fp16=accelerator.mixed_precision != 'fp32',
                fp_scaler=fp_scaler,
                accelerator=accelerator,
                grad_accum=False
            )
            
            step_loss_backward_partial()
            ema_net.update()

        # scheduler update
        # FIXME: not support transformers ReduceLROnPlateau which is LRLambda, may be using inspect can fix?
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step(loss)
        else:
            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step(ep)
            
        # eval
        if ep % eval_every_epochs == 0:
            model.eval()
            optimizer = _optimizer_eval(optimizer)
            with ema_net.average_parameters() and accelerator.autocast():
                val_acc_dict, val_loss = val(accelerator, model, val_dl, criterion, logger, ep, optim_val_loss, args)
            model.train()
            optimizer = _optimizer_train(optimizer)
        
            # @is_main_proces  # deep speed need all reduce
            # NOTE: exists hf state already, it is a need for saving model params manually?
            def collect_params():
                # TODO: ZERO-3 not save model
                # if accelerator.distributed_type == DistributedType.DEEPSPEED:
                #     return None
                params = {}
                # params["model"] = model_params(model)
                params["ema_model"] = ema_net.state_dict()  # TODO: contain on-the-fly params, find way to remove and not affect the load
                params["epochs"] = ep
                params["metrics"] = val_acc_dict
                
                return params
            
            params = collect_params()
            save_or_not = True
            if params is not None:
                if save_or_not := save_checker(val_acc_dict, val_loss, optim_val_loss):
                    # torch.save(params, save_path)
                    if accelerator.is_main_process:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        accelerator.save(params, save_path)
                        optim_val_loss = val_loss
                        logger.print("save EMA params")
            else:
                save_or_not = True
                
        if ep % args.checkpoint_every_n == 0 and save_or_not:
            logger.print(f'>>> save sate of {ep=}')
            output_dir = f"ep_{ep}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                
            # ref to deepspeed.runtime.engine.DeepSpeedEngine.save_checkpoint
            accelerator.save_state(output_dir)
            
        # print all info
        ep_loss /= i
        ep_loss_dict = ave_ep_loss(ep_loss_dict, i)
        lr = optimizer.param_groups[0]["lr"]
        if accelerator.use_distributed:
            ep_loss = accelerator.reduce(ep_loss, reduction='mean')   # sum
            ep_loss_dict = dist_gather_object(ep_loss_dict, n_ranks=accelerator.num_processes, dest=0)  # gather n_proc objs
            ep_loss_dict = ave_multi_rank_dict(ep_loss_dict)  # [{'l1': 0.1}, {'l1': 0.2}] -> {'l1': 0.15}

        if logger is not None and accelerator.use_distributed:
            if accelerator.is_main_process: 
                logger.log_curve(ep_loss, "train_loss", ep)
                for k, v in ep_loss_dict.items():
                    logger.log_curve(v, f'train_{k}', ep)
                logger.log_curve(lr, "lr", ep)
                logger.print(f"[{ep}/{epochs}] lr: {lr:.4e} " + loss_dict2str(ep_loss_dict))
        elif logger is None and accelerator.use_distributed:
            if accelerator.is_main_process:
                print(f"[{ep}/{epochs}] lr: {lr:.4e} " + loss_dict2str(ep_loss_dict))
        elif logger is not None and not accelerator.use_distributed:
            logger.log_curve(ep_loss, "train_loss", ep)
            for k, v in ep_loss_dict.items():
                logger.log_curve(v, f'train_{k}', ep)
            logger.log_curve(lr, "lr", ep)
            logger.print(f"[{ep}/{epochs}] lr: {lr:.4e} " + loss_dict2str(ep_loss_dict))
        else:
            print(f"[{ep}/{epochs}] lr: {lr:.4e} " + loss_dict2str(ep_loss_dict))

        # watch network params(grad or data or both)
        if isinstance(logger, TensorboardLogger):
            logger.log_network(model, ep)
