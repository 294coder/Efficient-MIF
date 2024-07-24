import argparse
import os
import os.path as osp

import h5py
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
import colored_traceback.always
# from rich.traceback import install
# install()
from wandb.util import generate_id

from torch_run_engine import train
from model import build_network
from utils import (
    TensorboardLogger,
    TrainProcessTracker,
    config_load,
    convert_config_dict,
    get_optimizer,
    get_scheduler,
    h5py_to_dict,
    is_main_process,
    merge_args_namespace,
    module_load,
    resume_load,
    BestMetricSaveChecker,
    set_all_seed,
    get_loss,
)


def get_main_args():
    parser = argparse.ArgumentParser("PANFormer")

    # network
    parser.add_argument("-a", "--arch", type=str, default="pannet")
    parser.add_argument("--sub_arch", default=None, help="panformer sub-architecture name")
    
    # train config
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--pretrain_id", type=str, default=None)
    parser.add_argument("--non_load_strict", action="store_false", default=True)
    parser.add_argument("-e", "--epochs", type=int, default=500)
    parser.add_argument("--val_n_epoch", type=int, default=30)
    parser.add_argument("--warm_up_epochs", type=int, default=80)
    parser.add_argument( "-l", "--loss", type=str, default="mse", choices=[ "mse", "l1", "hybrid", "smoothl1", "l1ssim", "charbssim", "ssimsf", "ssimmci", "mcgmci", "ssimrmi_fuse", "pia_fuse", "u2fusion", "swinfusion", "hpm", "none", "None",],)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--save_every_eval", action="store_true", default=False)

    # resume training config
    parser.add_argument("--resume_ep", default=None, required=False, help="do not specify it")
    parser.add_argument("--resume_lr", type=float, required=False, default=None)
    parser.add_argument("--resume_total_epochs", type=int, required=False, default=None)

    # path and load
    parser.add_argument("-p", "--path", type=str, default=None, help="only for unsplitted dataset")
    parser.add_argument("--split_ratio", type=float, default=None)
    parser.add_argument("--load", action="store_true", default=False, help="resume training")
    parser.add_argument("--save_base_path", type=str, default="./weight")

    # datasets config
    parser.add_argument("--dataset", type=str, default="wv3")
    parser.add_argument("-b", "--batch_size", type=int, default=1028)
    parser.add_argument("--hp", action="store_true", default=False)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--aug_probs", nargs="+", type=float, default=[0.0, 0.0])
    parser.add_argument("-s", "--seed", type=int, default=3407)
    parser.add_argument("-n", "--num_worker", type=int, default=8)
    parser.add_argument("--ergas_ratio", type=int, choices=[2, 4, 8, 16, 20], default=4)

    # logger config
    parser.add_argument("--logger_on", action="store_true", default=False)
    parser.add_argument("--proj_name", type=str, default="panformer_wv3")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument( "--resume", type=str, default="None", help="used in wandb logger, please not use it in tensorboard logger")
    parser.add_argument("--run_id", type=str, default=generate_id())
    parser.add_argument("--watch_log_freq", type=int, default=10)
    parser.add_argument("--watch_type", type=str, default="None")
    parser.add_argument("--metric_name_for_save", type=str, default="SAM")
    parser.add_argument("--log_metrics", action="store_true", default=False)

    # ddp setting
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--dp", action="store_true", default=False)
    parser.add_argument("--ddp", action="store_true", default=False)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("--fp16", action="store_true", default=False)

    # some comments
    parser.add_argument("--comment", type=str, required=False, default="")

    return parser.parse_args()


def main(local_rank, args):
    if type(local_rank) == str: 
        local_rank = int(local_rank.split(':')[-1])
    set_all_seed(args.seed + local_rank)
    torch.cuda.set_device(local_rank if args.ddp else args.device)

    # load other configuration and merge args and configs together
    configs = config_load(args.arch, "./configs")
    args = merge_args_namespace(args, convert_config_dict(configs))
    if is_main_process():
        print(args)

    # define network
    full_arch = args.arch + "_" + args.sub_arch if args.sub_arch is not None else args.arch
    args.full_arch = full_arch
    network_configs = getattr(
        args.network_configs, full_arch, args.network_configs
    ).to_dict()
    network = build_network(full_arch, **network_configs).cuda()
    # FIXME: may raise compile error
    # network = torch.compile(network)

    # parallel or not
    assert not (args.dp and args.ddp), "dp and ddp can not be True at the same time"
    if args.dp:
        network = nn.DataParallel(network, list(range(torch.cuda.device_count())), 0)
    elif args.ddp:
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=torch.cuda.device_count(),
            rank=local_rank,
        )
        network = network.to(local_rank)
        args.optimizer.lr *= args.world_size
        network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
        network = nn.parallel.DistributedDataParallel(
            network,
            device_ids=[local_rank],
            output_device=local_rank,
            # find_unused_parameters=True,
        )
    else:
        network = network.to(args.device)

    # optimization
    if args.grad_accum_steps is not None:
        lr_adjust_ratio = 1.0 #args.grad_accum_steps
    else:
        lr_adjust_ratio = 1.0
    args.optimizer.lr *= lr_adjust_ratio
    optim = get_optimizer(network.parameters(), **args.optimizer.to_dict())
    lr_scheduler = get_scheduler(optim, **args.lr_scheduler.to_dict())
    criterion = get_loss(args.loss, network_configs.get("spectral_num", 4)).cuda()

    # load params
    # assert not (args.load and args.resume == 'allow'), 'resume the network and wandb logger'
    status_tracker = TrainProcessTracker(id=args.run_id, resume=args.load, args=args)

    # pretrain load
    if args.pretrain:
        assert (
            args.pretrain_id is not None
        ), "you should specify @pretrain_id when @pretrain is True"
        p = osp.join(args.save_base_path, args.arch + "_" + args.pretrain_id + ".pth")
        network = module_load(
            p,
            network,
            args.device,
            local_rank if args.ddp else None,
            strict=args.non_load_strict,
        )
        if is_main_process():
            print("=" * 20, f"load pretrain weight id: {args.pretrain_id}", "=" * 20)

    # resume training
    # TODO: now this code clip is not used, try to clear the usage
    if args.load:
        args.resume = "allow"
        p = osp.join(
            args.save_base_path, args.arch + "_" + status_tracker.status["id"] + ".pth"
        )
        if args.resume_total_epochs is not None:
            assert (
                args.resume_total_epochs == args.epochs
            ), "@resume_total_epochs should equal to @epochs"
        network, optim, lr_scheduler, args.resume_ep = resume_load(
            p,
            network,
            optim,
            lr_scheduler,
            device=args.device,
            specific_resume_lr=args.resume_lr,
            specific_epochs=args.resume_total_epochs,
            ddp_rank=local_rank if args.ddp else None,
            ddp=args.ddp,
        )
        if is_main_process():
            print(f"load {p}", f"resume from epoch {args.resume_ep}", sep="\n")

    # get logger
    if is_main_process() and args.logger_on:
        # logger = WandbLogger(args.proj_name, config=args, resume=args.resume,
        #                      id=args.run_id if not args.load else status_tracker.status['id'], run_name=args.run_name)
        logger = TensorboardLogger(
            comment=args.run_id,
            args=args,
            file_stream_log=True,
            method_dataset_as_prepos=True,
        )
        logger.watch(
            network=network.module if args.ddp else network,
            watch_type=args.watch_type,
            freq=args.watch_log_freq,
        )
    else:
        from utils import NoneLogger
        logger = NoneLogger()

    # get datasets and dataloader
    if (
        args.split_ratio is not None and args.path is not None and False
    ):  # never reach here
        # FIXME: only support splitting worldview3 datasets
        # Warn: will be decrepated in the next update
        train_ds, val_ds = make_datasets(
            args.path,
            hp=args.hp,
            seed=args.seed,
            aug_probs=args.aug_probs,
            split_ratio=args.split_ratio,
        )
    else:
        if args.dataset == "flir":
            from task_datasets.RoadScene import RoadSceneDataset
            
            train_ds = RoadSceneDataset(args.path.base_dir, "train")
            val_ds = RoadSceneDataset(args.path.base_dir, "test")
        elif args.dataset == "tno":
            from task_datasets.TNO import TNODataset
        
            train_ds = TNODataset(
                args.path.base_dir, "train", aug_prob=args.aug_probs[0]
            )
            val_ds = TNODataset(args.path.base_dir, "test", aug_prob=args.aug_probs[1])

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
            assert train_path is not None and val_path is not None, "train_path and val_path should not be None"

            h5_train, h5_val = h5py.File(train_path), h5py.File(val_path),
            
            if args.dataset in ["wv3", "qb"]:
                from task_datasets.WV3 import WV3Datasets, make_datasets
                
                d_train, d_val = h5py_to_dict(h5_train), h5py_to_dict(h5_val)
                train_ds, val_ds = (
                    WV3Datasets(d_train, hp=args.hp, aug_prob=args.aug_probs[0]),
                    WV3Datasets(d_val, hp=args.hp, aug_prob=args.aug_probs[1]),
                )
            elif args.dataset == "gf2":
                from task_datasets.GF2 import GF2Datasets
                
                d_train, d_val = h5py_to_dict(h5_train), h5py_to_dict(h5_val)
                train_ds, val_ds = (
                    GF2Datasets(d_train, hp=args.hp, aug_prob=args.aug_probs[0]),
                    GF2Datasets(d_val, hp=args.hp, aug_prob=args.aug_probs[1]),
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
        
    # from torch.utils.data import Subset
    # train_ds = Subset(train_ds, range(0, 20))
        
    if args.ddp:
        train_sampler = torch.utils.data.DistributedSampler(train_ds, shuffle=args.shuffle)
        val_sampler = torch.utils.data.DistributedSampler(val_ds, shuffle=args.shuffle)
    else:
        train_sampler, val_sampler = None, None
    
    train_dl = data.DataLoader(
        train_ds,
        args.batch_size,
        num_workers=args.num_worker,
        sampler=train_sampler,
        prefetch_factor=8,
        pin_memory=True,
        shuffle=args.shuffle if not args.ddp else None,
    )
    val_dl = data.DataLoader(
        val_ds,
        1,
        # args.batch_size,
        num_workers=args.num_worker,
        sampler=val_sampler,
        pin_memory=True,
        shuffle=args.shuffle if not args.ddp else None,
    )

    # make save path legal
    if not args.save_every_eval:
        args.save_path = osp.join(
            args.save_base_path, args.arch + "_" + args.run_id + ".pth"
        )
        if not osp.exists(args.save_base_path):
            os.makedirs(args.save_base_path)
    else:
        args.save_path = osp.join(args.save_base_path, args.arch + "_" + args.run_id)
        if not osp.exists(args.save_path):
            os.makedirs(args.save_path)
    print("network params are saved at {}".format(args.save_path))

    # save checker
    save_checker = BestMetricSaveChecker(metric_name="PSNR", check_order="up")

    # start training
    with status_tracker:
        train(
            network,
            optim,
            criterion,
            args.warm_up_epochs,
            lr_scheduler,
            train_dl,
            val_dl,
            args.epochs,
            args.val_n_epoch,
            args.save_path,
            logger=logger,
            resume_epochs=args.resume_ep or 1,
            ddp=args.ddp,
            check_save_fn=save_checker,
            fp16=args.fp16,
            max_norm=args.max_norm,
            grad_accum_steps=args.grad_accum_steps,
            args=args,
        )

    # logger finish
    status_tracker.update_train_status("done")
    if is_main_process() and logger is not None:
        logger.writer.close()

if __name__ == "__main__":
    import torch.multiprocessing as mp

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5700"
    os.environ["HF_HOME"] = ".cache/transformers"
    os.environ["MPLCONFIGDIR"] = ".cache/matplotlib"

    args = get_main_args()
    
    if (not args.ddp) and (not args.dp): 
        print('>>> TORCH: using one gpu')
        main(args.device, args)
    else:
        print('>>> SPAWN: using multiple gpus')
        # TODO: resort to accelerator method, to avoid multi-processing dataloader using more RAM
        mp.spawn(main, args=(args,), nprocs=args.world_size if args.ddp else 1)
    
        
        
        