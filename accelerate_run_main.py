import os
import sys
from types import SimpleNamespace

os.environ["HF_HOME"] = ".cache/transformers"
os.environ["MPLCONFIGDIR"] = ".cache/matplotlib"
import argparse
import os.path as osp
from rich.console import Console

import time
import h5py
import torch
import torch.nn as nn
import torch.utils.data as data
import accelerate
from accelerate.utils import DummyOptim, DummyScheduler, set_seed

from accelerate_run_hyper_ms_engine import train as train_sharpening
from accelerate_run_VIS_IR_engine import train as train_fusion
from accelerate.utils import ProjectConfiguration
from model import build_network
from utils import (
    TensorboardLogger,
    config_load,
    convert_config_dict,
    get_optimizer,
    get_scheduler,
    h5py_to_dict,
    is_main_process,
    merge_args_namespace,
    module_load,
    BestMetricSaveChecker,
    get_loss,
    generate_id,
    LoguruLogger,
    get_fusion_dataset,
    EasyProgress
)

logger = LoguruLogger.logger(sink=sys.stdout)
LoguruLogger.add('log_file/running_traceback.log', format="{time:MM-DD hh:mm:ss} {level} {message}", 
                level="WARNING", backtrace=True, diagnose=True, mode='w')
LoguruLogger.add(sys.stderr, format="{time:MM-DD hh:mm:ss} {level} {message}", 
                level="ERROR", backtrace=True, diagnose=True)


def get_main_args():
    parser = argparse.ArgumentParser("PANFormer")

    # network
    parser.add_argument("-a", "--arch", type=str, default="pannet")
    parser.add_argument("--sub_arch", default=None, help="panformer sub-architecture name")
    
    # train config
    
    parser.add_argument("--pretrain_model_path", type=str, default=None, help='pretrained model path')
    parser.add_argument("--non_load_strict", action="store_false", default=True)
    parser.add_argument("-e", "--num_train_epochs", type=int, default=500)
    parser.add_argument("--val_n_epoch", type=int, default=30)
    parser.add_argument("--warm_up_epochs", type=int, default=10)
    parser.add_argument("-l", "--loss", type=str, default="mse")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--checkpoint_every_n", default=None, type=int, help='checkpointing the running state whether the saving condition is met or not '
                                                                             '(see the `check_save_fn` in the training function)')
    parser.add_argument("--ckpt_max_limit", type=int, default=10, help='maximum number of checkpoints to keep')
    parser.add_argument("--mix_precison", default='fp32', choices=['fp32', 'fp16'], help="mixed precision training")
    parser.add_argument("--sanity_check", action="store_true", default=False)
    # decrepted
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--pretrain_id", type=str, default=None)

    # resume training config
    parser.add_argument("--resume_path", default=None, required=False, help="path for resuming state")
    # decrepcted
    parser.add_argument("--resume_lr", type=float, required=False, default=None)
    parser.add_argument("--resume_total_epochs", type=int, required=False, default=None)
    parser.add_argument("--nan_no_raise", action="store_true", default=False, help='not raise error when nan loss')

    # path and load
    parser.add_argument("-p", "--path", type=str, default=None, help="only for unsplitted dataset")
    parser.add_argument("--split_ratio", type=float, default=None)
    parser.add_argument("--load", action="store_true", default=False, help="resume training")
    parser.add_argument("--save_base_path", type=str, default="./weight")

    # datasets config
    parser.add_argument("--dataset", type=str, default="wv3")
    parser.add_argument("-b", "--batch_size", type=int, default=1028, help='set train and val bs')
    parser.add_argument("--train_bs", type=int, default=None)
    parser.add_argument("--val_bs", type=int, default=None)
    parser.add_argument("--fusion_crop_size", type=int, default=72, help='image cropped size for fusion task')
    parser.add_argument("--only_y", action="store_true", default=False)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--fast_eval_n_samples", type=int, default=128)
    parser.add_argument("--aug_probs", nargs="+", type=float, default=[0.0, 0.0])
    parser.add_argument("-s", "--seed", type=int, default=3407)
    parser.add_argument("-n", "--num_worker", type=int, default=8)
    parser.add_argument("--ergas_ratio", type=int, choices=[2, 4, 8, 16, 20], default=4)

    # logger config
    parser.add_argument("--logger_on", action="store_true", default=False)
    parser.add_argument("--proj_name", type=str, default="panformer_wv3")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume_id", type=str, default="None", help='resume training id')
    parser.add_argument("--run_id", type=str, default=generate_id())
    parser.add_argument("--watch_log_freq", type=int, default=10)
    parser.add_argument("--watch_type", type=str, default="None")
    parser.add_argument("--metric_name_for_save", type=str, default="SAM")
    parser.add_argument("--log_metrics", action="store_true", default=False)

    # ddp setting
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--ddp", action="store_true", default=False)

    # some comments
    parser.add_argument("--comment", type=str, required=False, default="")

    return parser.parse_args()


def main(args):
    accelerator = accelerate.Accelerator(mixed_precision='no',
                                         gradient_accumulation_steps=args.grad_accum_steps,)
    set_seed(args.seed)
    print(f'>>> PID - {os.getpid()}: accelerate launching...')
    # logger.print(repr(args))
    
    device = accelerator.device
    # torch.cuda.set_device(device)
    args.device = str(device)
    args.ddp = accelerator.use_distributed
    args.world_size = accelerator.num_processes
    
    # get network config
    configs = config_load(args.arch, "./configs")
    args = merge_args_namespace(args, convert_config_dict(configs))
    
    # define network
    full_arch = args.arch + "_" + args.sub_arch if args.sub_arch is not None else args.arch
    args.full_arch = full_arch
    network_configs = getattr(args.network_configs, full_arch, args.network_configs).to_dict()
    network = build_network(full_arch, **network_configs).to(device)
        
    # get logger
    if args.logger_on and accelerator.is_main_process:
        logger = TensorboardLogger(
            comment=args.run_id,
            args=args,
            file_stream_log=True,
            method_dataset_as_prepos=True,
        )
        logger.watch(
            network=network, #.module if args.ddp else network,
            watch_type=args.watch_type,
            freq=args.watch_log_freq,
        )
    else:
        from utils import NoneLogger
        logger = NoneLogger(cfg=args, name=args.proj_name)
        
    # make save path legal
    
    # FIXME: temporary solution (handle the run_id) using ddp to exchange the run_id
    if accelerator.is_main_process:
        # args.output_dir = args.save_base_path = osp.join(args.save_base_path, 
        #                                                  time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
        #                                                 + "_" +args.arch + "_" + args.run_id)
        # args.save_model_path = osp.join(args.save_base_path, args.arch + "_" + args.run_id + '.pth')
        
        weight_path = osp.join(logger.log_file_dir, 'weights')
        if accelerator.is_main_process:
            os.makedirs(weight_path, exist_ok=True)
        args.output_dir = weight_path
        args.save_model_path = osp.join(args.output_dir, 'ema_model.pth')
    else:
        # for ddp broadcast
        args.output_dir = args.save_model_path = args.save_base_path = [None] 
        
    if accelerator.use_distributed:
        (args.output_dir, 
         args.save_model_path,
         args.save_base_path) = accelerate.utils.broadcast_object_list([args.output_dir, 
                                                                        args.save_model_path, 
                                                                        args.save_base_path])
    
    
    accelerator.project_configuration = project_config = ProjectConfiguration(project_dir=args.output_dir, 
                                                                            total_limit=args.ckpt_max_limit,)
    logger.info(f'model weights will be save at {args.output_dir}')
            
    # handle the optimizer and lr_scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in network.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.optimizer.weight_decay,},{
            "params": [p for n, p in network.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # pretrain load
    # if args.pretrain:
    #     assert (args.pretrain_id is not None), "you should specify @pretrain_id when @pretrain is True"
    #     args.output_dir = osp.join(args.save_base_path, args.arch + "_" + args.pretrain_id)
    #     p = osp.join(args.output_dir, "best_model.pth")
    #     network = module_load(p, network, device, device if args.ddp else None,
    #                           strict=args.non_load_strict,)
    #     logger.print("=" * 20, f"load pretrain weight id: {args.pretrain_id}", "=" * 20)

    
    # get loss and dataset
    loss_cfg = getattr(args, 'loss_cfg', SimpleNamespace())
    if loss_cfg is None:
        loss_cfg = SimpleNamespace()
    loss_cfg.fusion_model = network
    loss_cfg.device = device
    loss_cfg.model_is_y_pred = args.only_y
    criterion = get_loss(args.loss,
                         network_configs.get("spectral_num", 3),
                         **vars(loss_cfg)).to(device)
    
    if args.train_bs is None:
        args.train_bs = args.batch_size
    if args.val_bs is None:
        args.val_bs = args.batch_size
    
    # get train and val dataset and dataloader
    train_ds, train_dl, val_ds, val_dl = get_fusion_dataset(args, accelerator, device)
        
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    if (accelerator.state.deepspeed_plugin is None 
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config):
        optimizer = get_optimizer(network, network.parameters(), **args.optimizer.to_dict())
    else:
        optimizer = DummyOptim(optimizer_grouped_parameters, lr=args.learning_rate)
        
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (accelerator.state.deepspeed_plugin is None 
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config):
        # transformers lr scheduler
        # from transformers import get_scheduler
        # lr_scheduler = get_scheduler(
        #     name=args.lr_scheduler_type,
        #     optimizer=optimizer,
        #     num_warmup_steps=args.num_warmup_steps,
        #     num_training_steps=args.max_train_steps,
        # )
        lr_scheduler = get_scheduler(optimizer, **args.lr_scheduler.to_dict())
    else:
        # a placeholder
        lr_scheduler = DummyScheduler(optimizer, 
                                      total_num_steps=args.num_train_epochs, 
                                      warmup_num_steps=args.warm_up_epochs)
    
    logger.print("network params and training states are saved at [dim green]{}[/dim green]".format(args.save_model_path))
    
    
    # save checker and train process tracker
    save_checker = BestMetricSaveChecker(metric_name=args.metric_name_for_save)
    # status_tracker = TrainProcessTracker(id=args.run_id, resume=args.load, args=args)
    
    # start training
    # with status_tracker:
    train_fn = train_fusion if args.task == "fusion" else train_sharpening
    train_fn(
        accelerator,
        network,
        optimizer,
        criterion,
        args.warm_up_epochs,
        lr_scheduler,
        train_dl,
        val_dl,
        args.num_train_epochs,
        args.val_n_epoch,
        args.save_model_path,
        logger=logger,
        resume_epochs=1,
        ddp=args.ddp,
        check_save_fn=save_checker,
        max_norm=args.max_norm,
        args=args,
    )
        
    # logger finish
    # status_tracker.update_train_status("done")
    if is_main_process() and logger is not None:
        logger.writer.close()
        
if __name__ == "__main__":
    args = get_main_args()
    
    try:
        main(args)
    except Exception as e:
        EasyProgress.close_all_tasks()
        logger.error('An Error Occurred! Please check the stacks in log_file/running_traceback.log')
        logger.exception(e)
        # raise e