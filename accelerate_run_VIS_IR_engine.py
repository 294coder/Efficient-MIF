import gc
import os
import math
from pathlib import Path
from typing import Callable, Union

import accelerate.scheduler
import torch
from torch.utils.data import DataLoader
import accelerate
from accelerate.utils import DistributedType

import warnings
# filter off all warnings
warnings.filterwarnings("ignore", module="torch")

from model.base_model import BaseModel
from utils import (
    AnalysisPanAcc,
    AnalysisVISIRAcc,
    dict_to_str,
    prefixed_dict_key,
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
    sanity_check,
    y_pred_model_colored,
    EMA,
    WindowBasedPadder,
)
from schedulefree import AdamWScheduleFree
from utils.log_utils import TensorboardLogger
from typing_extensions import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
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

def get_analyser(args):
    if args.log_metrics:
        # vis-ir fusion task
        if args.task == 'fusion':
            analyser = AnalysisVISIRAcc()
        else:
            analyser = AnalysisPanAcc(args.ergas_ratio)
    else:
        analyser = NonAnalysis()
    
    return analyser


@torch.inference_mode()
def val(
        accelerator: accelerate.Accelerator,
        network: BaseModel,
        val_dl: DataLoader,
        criterion: Callable,
        logger: Union[TensorboardLogger],
        ep: int = None,
        optim_val_loss: float = None,
        args=None,
):
    torch.cuda.empty_cache()  # may be useful
    gc.collect()
    
    val_loss = 0.0
    val_loss_dict = {}
    
    # get analysor for validate metrics
    analysis = get_analyser(args)
    padder = WindowBasedPadder(32)
    
    dtype = get_precision(accelerator.mixed_precision)
    tbar, task_id = EasyProgress.easy_progress(["Validation"], [len(val_dl)], 
                                               is_main_process=accelerator.is_main_process,
                                               tbar_kwargs={'console': logger.console})
    if accelerator.is_main_process:
        tbar.start()
    logger.print('=======================================EVAL STAGE=================================================')
    for i, (vis, ir, mask, gt) in enumerate(val_dl, 1):
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            vis = vis.to(accelerator.device, dtype=dtype)
            ir = ir.to(accelerator.device, dtype=dtype)
            gt = gt.to(accelerator.device, dtype=dtype)
            
        if mask is not None:
            if isinstance(mask, torch.Tensor) and mask.ndim > 2:
                mask = mask.to(accelerator.device, dtype=dtype)
            else:
                mask = None

        ir = padder(ir)
        vis = padder(vis, no_check_pad=True)
        with y_pred_model_colored(vis, enable=args.only_y) as (vis_y, back_to_rgb):
            outp = network(vis_y, ir, mask, mode="fusion_eval", to_rgb_fn=back_to_rgb)
            # outp = back_to_rgb(outp)
        
        if isinstance(outp, (tuple, list)):
            fused, seg_map = outp
        else:
            assert torch.is_tensor(outp), 'output should be a tensor'
            fused = outp
            seg_map = None

        fused = padder.inverse(fused)
        # TODO: handle the segmap?
                       
        loss_out = criterion(fused, gt, mask=mask)
        # if loss is hybrid, will return tensor loss and a dict
        if isinstance(loss_out, tuple):
            batched_loss, loss_d = loss_out
        else:
            batched_loss = loss_out
            loss_d = {'val_main_loss': loss_out}

        analysis(gt, fused)
        val_loss += batched_loss
        val_loss_dict = accum_loss_dict(val_loss_dict, loss_d)
        
        # advance the task_id
        if accelerator.is_main_process and task_id is not None:
            tbar.update(task_id, total=len(val_dl), completed=i, visible=True if i < len(val_dl) else False,
                        description=f'Validation iter [{i}/{len(val_dl)}] - {loss_dict2str(loss_d)}')
    
    logger.print(analysis.result_str(), dist=True, proc_id=accelerator.process_index)  # log in every process

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
                vis, ir, fused= list(map(func, [vis, ir, fused]))
            logger.log_images([vis, ir, fused], nrow=4, names=["vis", "ir", "fused"],
                              epoch=ep, task=args.task, ds_name=args.dataset)

        # print eval info
        logger.info('\n\nsummary of evaluation:')
        logger.info(f'evaluate {n} samples')
        logger.info(loss_dict2str(val_loss_dict))
        logger.info(f"\n{dict_to_str(acc_ave)}" if args.log_metrics else "")
        logger.info('==================================================================================================')
    return acc_ave, val_loss  # only rank 0 is reduced and other ranks are original data


def train(
        accelerator: accelerate.Accelerator,
        model: BaseModel,
        optimizer,
        criterion,
        warm_up_epochs,
        lr_scheduler,
        train_dl: "DataLoader | DALIGenericIterator" ,
        val_dl: "DataLoader | DALIGenericIterator",
        epochs: int,
        eval_every_epochs: int,
        save_path: str,
        check_save_fn: Callable=None,
        logger: Union[TensorboardLogger] = None,
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
    # check save function
    save_checker = lambda *check_args: check_save_fn(check_args[0]) if check_save_fn is not None else \
                   lambda val_acc_dict, val_loss, optim_val_loss: val_loss < optim_val_loss
    
    dtype = get_precision(accelerator.mixed_precision)
    
    # load pretrain model
    if args.pretrain_model_path is not None:
        # e.g., panMamba.pth
        assert args.pretrain_model_path.endswith('.pth') or args.pretrain_model_path.endswith('.safetensors')
        model = module_load(args.pretrain_model_path, model, device=accelerator.device, strict=args.non_load_strict)
    
    # Prepare everything with accelerator
    model, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(model, optimizer, train_dl, val_dl, lr_scheduler)
    
    # FIXME: Deepspeed ZERO3 does not support!
    # ema_net = ExponentialMovingAverage(parameters=[p for p in model.parameters() if p.requires_grad],
    #                                    decay=args.ema_decay)
    # ema_net.to(accelerator.device)
    ema_net = EMA(model, beta=args.ema_decay, update_every=2)
    accelerator.register_for_checkpointing(ema_net)
    
    # load state
    if args.resume_path is not None:
        # e.g., panMamba/**/ep_80
        accelerator.load_state(input_dir=args.resume_path)
        logger.info(f">>> PROCESS {accelerator.process_index}: loaded state from {args.resume_path} done.")
    
    accelerator.wait_for_everyone()
    if args.sanity_check:
        logger.print(">>> sanity check...")
        with torch.no_grad():
            sanity_check_val = sanity_check(val)
            sanity_check_out = sanity_check_val(accelerator, model, val_dl, criterion, logger, 0, torch.inf, args)
            torch.cuda.empty_cache()
            gc.collect()

    # Figure out how many steps we should save the Acclerator states
    # checkpointing_steps = args.checkpointing_steps
    # if checkpointing_steps is not None and checkpointing_steps.isdigit():
    #     checkpointing_steps = int(checkpointing_steps)
    
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
    for ep in range(resume_epochs, epochs + 1):
        ep_loss = 0.0
        ep_loss_dict = {}
        i = 0
        
        # model training
        for i, (vis, ir, mask, gt) in enumerate(train_dl, 1):
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                vis = vis.to(accelerator.device, dtype=dtype)
                ir = ir.to(accelerator.device, dtype=dtype)
                gt = gt.to(accelerator.device, dtype=dtype)
                
            if mask is not None:
                if isinstance(mask, torch.Tensor) and mask.ndim > 2:
                    mask = mask.to(accelerator.device, dtype=dtype)
                else:
                    mask = None
            
            # model get data and compute loss
            with accelerator.autocast() and accelerator.accumulate(model):
                with y_pred_model_colored(vis, enable=args.only_y) as (vis_y, back_to_rgb):
                    fused, loss_out = model(vis_y, ir, mask, gt, criterion, 
                                            mode="fusion_train", to_rgb_fn=back_to_rgb,
                                            has_gt=args.has_gt)  # hack the model the first arg
                    # fused = back_to_rgb(fused)
            
            # if loss is hybrid, will return tensor loss and a dict
            if isinstance(loss_out, (tuple, list)):
                loss, loss_d = loss_out
            else: 
                assert isinstance(loss_out, torch.Tensor), 'loss should be a tensor'
                loss = loss_out
            
            # if accelerator.sync_gradients:
            #     completed_steps += 1
                    
            ep_loss += loss
            
            # check nan loss
            if args.nan_no_raise:
                _grouped_loss = accelerator.gather(loss)
                if torch.isnan(_grouped_loss).any():
                    # 2. re-use the previous checkpoints
                    outp = Path(args.output_dir)
                    if outp.exists():
                        # sort it and use the latest
                        ckpts = outp.glob('ep*')
                        ckpts = sorted(ckpts, key=lambda x: int(x.name.split('_')[-1]))
                        accelerator.load_state(ckpts[-1])
                        
                        logger.info(f">>> PROCESS {accelerator.process_index}: loss is nan, re-use the previous checkpoints")
                    else:
                        logger.error(f">>> PROCESS {accelerator.process_index}: loss is nan", 
                                     raise_error=True, error_type=ValueError)  # raise error
            else:
                assert not torch.isnan(loss), f"loss is nan at process={accelerator.process_index}, batch_idx={i}, epoch={ep}"
            
            # update parameters
            step_loss_backward(
                optim=optimizer,
                network=model,
                max_norm=max_norm,
                loss=loss,
                fp16=accelerator.mixed_precision != 'fp32',
                fp_scaler=fp_scaler,
                accelerator=accelerator,
                grad_accum=False
            )
            if accelerator.sync_gradients:
                ema_net.update()
            
            # update the progress bar
            ep_loss_dict = accum_loss_dict(ep_loss_dict, loss_d)
            if accelerator.is_main_process:
                tbar.update(iter_task, total=len(train_dl), completed=i, visible=True,
                            description=f'Training iter [{i}/{len(train_dl)}] - {loss_dict2str(loss_d)}')
        
        # scheduler update
        # FIXME: not support transformers ReduceLROnPlateau which is LRLambda, may be using inspect can fix?
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step(loss)
        else:
            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step(ep)
            else:
                logger.info('>>> optimizer step was skipped due to mixed precision overflow')
            
        # eval
        if (ep % eval_every_epochs == 0) and (eval_every_epochs != -1):
            ema_net.eval()
            optimizer = _optimizer_eval(optimizer)
            with accelerator.autocast():
                val_acc_dict, val_loss = val(accelerator, ema_net.ema_model, val_dl,
                                             criterion, logger, ep, optim_val_loss, args)
                torch.cuda.empty_cache()  # may be useful
                gc.collect()
            model.train()
            optimizer = _optimizer_train(optimizer)
            
            # save ema model
            if not args.regardless_metrics_save:
                save_check_flag = save_checker(val_acc_dict, val_loss, optim_val_loss) 
            else:
                save_check_flag = True
            if save_check_flag and accelerator.is_main_process:
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
        
        # no validation, save ema model params per checkpoint_every_n
        if args.checkpoint_every_n is None and eval_every_epochs == -1:
            logger.info('>>> [blue]EMA[/blue] - no validation, save ema model params')
            ema_params = accelerator.unwrap_model(ema_net.ema_model).state_dict()
            accelerator.save(ema_params, save_path)
            
        # ep_loss average  
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

        # print all info
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
            
            
            
        
        
        
        
        