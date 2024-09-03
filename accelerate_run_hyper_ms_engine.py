import os
from functools import partial
import math
from typing import Union

import torch
import accelerate
from tqdm import tqdm
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
    EasyProgress,
    EMA,
    TensorboardLogger,
)

from schedulefree import AdamWScheduleFree
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
    if args.task == 'sharpening':
        analysis = AnalysisPanAcc(args.ergas_ratio)
    else:
        analysis = NonAnalysis()
    
    dtype = get_precision(accelerator.mixed_precision)
    tbar, task_id = EasyProgress.easy_progress(["Validation"], [len(val_dl)], 
                                               is_main_process=accelerator.is_main_process,
                                               tbar_kwargs={'console': logger.console})
    if accelerator.is_main_process:
        tbar.start()
    logger.print('=======================================EVAL STAGE=================================================')
    for i, (pan, ms, lms, gt) in enumerate(val_dl, 1):
        
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
        
        # advance the task_id
        if accelerator.is_main_process and task_id is not None:
            tbar.update(task_id, total=len(val_dl), completed=i, visible=True if i < len(val_dl) else False,
                        description=f'Validation iter [{i}/{len(val_dl)}] - {loss_dict2str(loss_d)}')

    val_loss /= i
    val_loss_dict = ave_ep_loss(val_loss_dict, i)
    
    if args.log_metrics:  # gather from all procs to proc 0
        mp_analysis = dist_gather_object(analysis, n_ranks=accelerator.num_processes)
    gathered_val_dict = dist_gather_object(val_loss_dict, n_ranks=accelerator.num_processes)
    val_loss_dict = ave_multi_rank_dict(gathered_val_dict)

    # log validation results
    if args.log_metrics:
        if accelerator.is_main_process and args.ddp:
            for a in mp_analysis:
                logger.info(a.result_str())  # log in every process
        elif not args.ddp:
            logger.info(analysis.result_str())

    # gather metrics and log image
    acc_ave = analysis.acc_ave
    if accelerator.is_main_process:
        if args.ddp and args.log_metrics:
            n = 0
            acc = analysis.empty_acc
            for a in mp_analysis:
                for k, v in a.acc_ave.items():
                    acc[k] += v * a._call_n
                n += a._call_n
            for k, v in acc.items():
                acc[k] = v / n
            acc_ave = acc
        else:
            n = analysis._call_n
            
        # if logger is not None:
        #     # log validate curves
        #     if args.log_metrics:
        #         logger.log_curves(prefixed_dict_key(acc_ave, "val"), ep)
        #     logger.log_curve(val_loss, "val_loss", ep)
        #     for k, v in val_loss_dict.items():
        #         logger.log_curve(v, f'val_{k}', ep)

        #     # log validate image(last batch)
        #     if gt.shape[0] > 8:
        #         func = lambda x: x[:8]
        #         gt, lms, pan, sr = list(map(func, [gt, lms, pan, sr]))
        #     residual_image = res_image(gt, sr, exaggerate_ratio=100)  # [b, 1, h, w]
        #     logger.log_images([lms, pan, sr, residual_image], nrow=4, 
        #                       names=["lms", "pan", "sr", "res"], 
        #                       epoch=ep, task='sharpening', 
        #                       ds_name=args.dataset)

        # print eval info
        logger.info('\n\nsummary of evaluation:')
        logger.info(f'evaluate {n} samples')
        logger.info(loss_dict2str(val_loss_dict))
        logger.info(f"\n{dict_to_str(acc_ave)}" if args.log_metrics else "")
        logger.info('==================================================================================================')
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
        check_save_fn: callable=None,
        logger: "TensorboardLogger" = None,
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
    :param args: other args, see more in main.py
    :return:
    """    
    save_checker = lambda *check_args: check_save_fn(check_args[0]) if check_save_fn is not None else \
                   lambda val_acc_dict, val_loss, optim_val_loss: val_loss < optim_val_loss

    dtype = get_precision(accelerator.mixed_precision)
    
     # load pretrain model
    if args.pretrain_model_path is not None:
        # e.g., panMamba.pth
        assert args.pretrain_model_path.endswith('.pth') or args.pretrain_model_path.endswith('.safetensors')
        model = module_load(args.pretrain_model_path, model, device=accelerator.device, 
                            strict=args.non_load_strict, spec_key='shadow_params')
    
    # Prepare everything with accelerator
    # and lr_scheduler not in prepare, or when multiprocessing the scheduler milestones will be changed
    model, optimizer, train_dl, val_dl = accelerator.prepare(
        model, optimizer, train_dl, val_dl
    )
    accelerator.register_for_checkpointing(lr_scheduler)
    
    # load state
    if args.resume_path is not None:
        # e.g., panMamba/**/ep_80
        accelerator.load_state(input_dir=args.resume_path)
        logger.info('>>> load state from {}'.format(args.resume_path))
    
    # FIXME: ZERO3 does not support!
    # ema_net = ExponentialMovingAverage(parameters=[p for p in model.parameters() if p.requires_grad],
    #                                    decay=args.ema_decay)
    # ema_net.to(next(model.parameters()).device)
    ema_net = EMA(model, beta=args.ema_decay, update_every=1, state_include_online_model=True)
    accelerator.register_for_checkpointing(ema_net)

    world_size = args.world_size if ddp else None
    optim_val_loss = math.inf
    fp_scaler = None  # accelerator.scaler if accelerator.mixed_precision != 'fp32' else None
    
    logger.print(f">>> start training!")
    logger.print(f">>> Num Iterations per Epoch = {len(train_dl)}")
    logger.print(f">>> Num Epochs = {args.num_train_epochs}")
    logger.print(f">>> Gradient Accumulation steps = {args.grad_accum_steps}")
    
    optimizer = _optimizer_train(optimizer)
    
    # handle the progress bar
    tbar, (ep_task, iter_task) = EasyProgress.easy_progress(["Epoch", "Iteration"], [epochs, len(train_dl)],
                                                            is_main_process=accelerator.is_main_process,
                                                            tbar_kwargs={'console': logger.console})
    if accelerator.is_main_process:
        tbar.start()
    accelerator.wait_for_everyone()
    
    for ep in range(resume_epochs, epochs + 1):
        ep_loss = 0.0
        ep_loss_dict = {}
        i = 0
        # model training
        for i, (pan, ms, lms, gt) in enumerate(train_dl, 1):
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
            
            if torch.isnan(loss).any():
                raise ValueError(f">>> PROCESS {accelerator.process_index}: loss is nan")

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
            
            # update the progress bar
            ep_loss_dict = accum_loss_dict(ep_loss_dict, loss_d)
            if accelerator.is_main_process:
                tbar.update(iter_task, total=len(train_dl), completed=i, visible=True,
                            description=f'Training iter [{i}/{len(train_dl)}] - {loss_dict2str(loss_d)}')

        # scheduler update
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step(loss)
        else:
            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step(ep)
            
        # eval
        if ep % eval_every_epochs == 0:
            optimizer = _optimizer_eval(optimizer)
            with accelerator.autocast():
                ema_net.eval()
                val_acc_dict, val_loss = val(accelerator, ema_net, val_dl, criterion, logger, ep, optim_val_loss, args)
            optimizer = _optimizer_train(optimizer)
            
            # save ema model
            if save_checker(val_acc_dict, val_loss, optim_val_loss) and accelerator.is_main_process:
                params = accelerator.unwrap_model(ema_net.ema_model).state_dict()

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                accelerator.save(params, save_path)
                optim_val_loss = val_loss
                logger.print(f">>> [green](validation)[/green] {ep=} - save EMA params")
        
        # checkpointing the running state
        checkpoint_flag = False if args.checkpoint_every_n is None \
                            else ep % args.checkpoint_every_n == 0
        if checkpoint_flag:
            logger.print(f'>>> [red](checkpoint)[/red] {ep=} - save training state')
            output_dir = f"ep_{ep}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            
            # save training state
            accelerator.save_state(output_dir)
                
        # set train mode
        model.train()
        accelerator.wait_for_everyone()
            
        # print all info
        ep_loss /= i
        ep_loss_dict = ave_ep_loss(ep_loss_dict, i)
        lr = optimizer.param_groups[0]["lr"]
        if accelerator.use_distributed:
            ep_loss = accelerator.reduce(ep_loss, reduction='mean')   # sum
            ep_loss_dict = dist_gather_object(ep_loss_dict, n_ranks=accelerator.num_processes, dest=0)  # gather n_proc objs
            ep_loss_dict = ave_multi_rank_dict(ep_loss_dict)  # [{'l1': 0.1}, {'l1': 0.2}] -> {'l1': 0.15}

        # advance the progress bar
        if accelerator.is_main_process:
            tbar.reset(iter_task)
            tbar.update(ep_task, total=epochs, completed=ep, visible=True,
                        description=f'Epoch [{ep}/{epochs}] - ep_loss: {loss_dict2str(ep_loss_dict)}')
        
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
