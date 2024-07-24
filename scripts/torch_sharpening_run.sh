## ddp training manner
#CUDA_VISIBLE_DEVICES=0,12 torchrun --nproc_per_node=2 --nnode=12\
# --master_port=23456 main.py --ddp --proj_name pannet --run_name run1 \
# --resume 'None' --epochs 1000 --init_lr 1e-3 --end_lr 5e-5 --hp

## dp train
#python main.py --proj_name pannet --arch pannet \
#  -b 512 -d 'cuda:12' --ddp --num_worker 6 \
#  --epochs 800 --warm_up_epochs 50 --val_n_epoch 20 \
#  --wandb_on --comment 'test tensorboard'
#  --load --resume "allow" --run_id "11pp4k7x"
#  --fp16 --hp

## panformer arch
#python main.py --proj_name panformer --arch fusionnet \
#  -b 1024 --device 'cuda:12' \
#  --warm_up_epochs 100 --num_worker 6 -e 1500 --ddp \
#  --loss mse --val_n_epoch 50 #--run_id None #--wandb_on #--load --resume "allow" --run_id "2uqcjdku"

## DCFNet arch
#python main.py --proj_name dcfnet --arch dcfnet \
#-b 64 --dataset 'hisi' --warm_up_epochs 0 --num_worker 6 -e 1500 --aug_probs 0. 0. \
#--logger_on --log_metrics --device 'cuda:1' --loss l1ssim --val_n_epoch 10 --ergas_ratio 4 \
#--comment '在新数据上训练dcfnet（wv3数据集）'
#--pretrain --pretrain_id 2g9q0m22

## lformer arch
# python main.py --proj_name lformer --arch lformer \
# -b 64 --device 'cuda:1' --dataset 'gf2' \
# --warm_up_epochs 0 --num_worker 6 -e 2000 --aug_probs 0. 0. \
# --loss l1ssim --val_n_epoch 10 --comment 'lformer arch without reduce on gf2 dataset' \
# --logger_on --log_metrics \
# --pretrain --pretrain_id '3tytop4x' #--non_load_strict


## MIMO_SST arch
python main.py --proj_name MIMO_SST --arch MIMO_SST \
-b 32 --device 'cuda:0' --dataset 'wv3' \
--warm_up_epochs 10 --num_worker 6 -e 2000 --aug_probs 0. 0. \
--loss l1ssim --val_n_epoch 20 --comment 'MIMO_SST reduced arch on wv3 dataset' \
--log_metrics \
--logger_on \
# --pretrain --pretrain_id 'd02s63w5'

## panMamba_zhouman arch
# python main.py --proj_name panmamba_zhouman --arch panmamba_zhouman \
# -b 128 --device 'cuda:1' --dataset 'wv3' \
# --warm_up_epochs 0 --num_worker 6 -e 2000 --aug_probs 0. 0. \
# --loss l1ssim --val_n_epoch 20 --comment 'panmamba_zhouman trained on wv3 dataset' \
# --logger_on --log_metrics \

## panMamba arch
# python main.py --proj_name panMamba --arch panMamba \
# -b 22 --device 'cuda:2' --dataset 'gf2' \
# --warm_up_epochs 0 --num_worker 6 -e 2000 --aug_probs 0. 0. \
# --loss l1ssim --val_n_epoch 20 --comment 'panMamba small config on gf2 dataset' \
# --logger_on --log_metrics \
# --pretrain --pretrain_id '39j3653x' --non_load_strict

## dcformer arch
# python main.py --proj_name lformer_eccv --arch lformer \
# -b 64 --device 'cuda:0' --dataset 'wv3' --logger_on --log_metrics \
# --warm_up_epochs 0 --num_worker 0 -e 2000 --aug_probs 0. 0. \
# --loss 'l1ssim' --val_n_epoch 10 --comment 'lformer on WV3 dataset' \
# --ergas_ratio 4
# --pretrain --pretrain_id '1x6ucirh' \
# --save_every_eval
# --non_load_strict

## dcformer_mwsa arch
# python main.py --proj_name dcformer --arch pmacnet \
# -b 5 --device 'cuda:1' --dataset 'qb' --log_metrics \
# --warm_up_epochs 0 --num_worker 0 -e 2000 --aug_probs 0. 0. \
# --loss 'l1ssim' --val_n_epoch 20 --logger_on --ergas_ratio 4 \
# --comment 'pmacnet on gf2'
# --pretrain --pretrain_id '1zlhpoze' --non_load_strict

## reciprocal_transformer arch
# python main.py --proj_name dct --arch dct \
# -b 8 --device 'cuda:1' --dataset 'cave_x8' \
# --warm_up_epochs 0 --num_worker 6 -e 2000 --aug_probs 0. 0. \
# --loss l1ssim --val_n_epoch 20 --comment 'reciprocal_transformer on cave_x8 dataset' \
# --logger_on --log_metrics \

## fuseformer arch
#  python main.py --proj_name dcformer --arch fuseformer\
#  -b 10 --device 'cuda:12' --dataset 'hisi' --logger_on \
#  --warm_up_epochs 30 --num_worker 0 -e 700 --aug_probs 0. 0. \
#  --loss l1 --val_n_epoch 10 --comment '训练fuseformer harvard' \
#  --pretrain --pretrain_id ufsb66w3

## hypert arch
#  python main.py --proj_name dcformer --arch hypertransformer \
#  -b 64 --device 'cuda:12' --dataset 'hisi' \
#  --warm_up_epochs 30 --num_worker 0 -e 500 --aug_probs 0. 0. \
#  --logger_on \
#  --loss l1 --val_n_epoch 20 --comment 'train hypertransformer on cave dataset' \
#  --pretrain --pretrain_id 2cmv1pb5

## Resume training script
#  python main.py --proj_name dcformer --arch dcformer --sub_arch 'reduce' \
#  -b 16 --device 'cuda:12' --dataset 'wv3' \
#  --warm_up_epochs 10 --num_worker 6 -e 600 --aug_probs 0. 0. \
#  --ddp --logger_on --load --resume_lr 1e-4 --resume_total_epochs 600 \
#  --loss mse --val_n_epoch 20 \
#  --comment 'dcformer_reduce on wv3 with less depth [4, [4, 3], [4, 3, 2], resume training' \
#  --run_id "lvu3ts9m"

#python main.py --proj_name panformer --arch panformer --sub_arch sga \
#  -b 8 --device 'cuda:12' \
#  --warm_up_epochs 10 --num_worker 6 -e 200 \
#  --ddp --wandb_on \
#  --loss mse --val_n_epoch 15 --comment 'test tensorboard logger'
#--wandb_on \

# fusionnet
#python main.py --proj_name panformer --arch fusionnet \
#  -b 512 --ddp --num_worker 6 \
#  --resume 'None' --epochs 800 --loss mse --wandb_on


## test patch_merge_module
#python model/dcformer_reduce.py

## ddp training manner
#CUDA_VISIBLE_DEVICES=0,12 torchrun --nproc_per_node=2 --nnode=12\
# --master_port=23456 main.py --ddp --proj_name pannet --run_name run1 \
# --resume 'None' --epochs 1000 --init_lr 1e-3 --end_lr 5e-5 --hp

## dp train
#python main.py --proj_name pannet --arch pannet \
#  -b 512 -d 'cuda:12' --ddp --num_worker 6 \
#  --epochs 800 --warm_up_epochs 50 --val_n_epoch 20 \
#  --wandb_on --comment 'test tensorboard'
#  --load --resume "allow" --run_id "11pp4k7x"
#  --fp16 --hp

## panformer arch
#python main.py --proj_name panformer --arch fusionnet \
#  -b 1024 --device 'cuda:12' \
#  --warm_up_epochs 100 --num_worker 6 -e 1500 --ddp \
#  --loss mse --val_n_epoch 50 #--run_id None #--wandb_on #--load --resume "allow" --run_id "2uqcjdku"

## DCFNet arch
#python main.py --proj_name dcfnet --arch dcfnet \
#-b 64 --dataset 'hisi' --warm_up_epochs 0 --num_worker 6 -e 1500 --aug_probs 0. 0. \
#--logger_on --log_metrics --device 'cuda:1' --loss l1ssim --val_n_epoch 10 --ergas_ratio 4 \
#--comment '在新数据上训练dcfnet（wv3数据集）'
#--pretrain --pretrain_id 2g9q0m22

## mmnet arch
#  python main.py --proj_name mmnet --arch mmnet \
#  -b 40 --device 'cuda:12' --dataset 'wv3' \
#  --warm_up_epochs 30 --num_worker 0 -e 500 --aug_probs 0. 0. \
#  --loss l1 --val_n_epoch 10 --comment '训练wv3 mmnet' \
#  --logger_on

## pannet arch
#  python main.py --proj_name pannet --arch pannet \
#  -b 60 --device 'cuda:0' --dataset 'qb' \
#  --warm_up_epochs 0 --num_worker 0 -e 600 --aug_probs 0. 0. \
#  --loss mse --val_n_epoch 10 --comment 'pannet qb' \
#  --logger_on

## dcformer arch
#  python main.py --proj_name dcformer --arch dcformer --sub_arch 'reduce' \
#  -b 30 --device 'cuda:12' --dataset 'qb' \
#  --warm_up_epochs 30 --num_worker 0 -e 600 --aug_probs 0. 0. \
#  --loss l1 --val_n_epoch 10 --comment '在大batch上训练 qb 调整了初始化 改变了head的数量 增加了dropout' \
#  --logger_on
#  --pretrain --pretrain_id '2oy4iqpj'

## dcformer_mwsa arch
# python main.py --proj_name lformer --arch lformer \
# -b 64 --device 'cuda:0' --dataset 'wv3' --logger_on --log_metrics \
# --warm_up_epochs 0 --num_worker 0 -e 2000 --aug_probs 0. 0. \
# --loss 'l1ssim' --val_n_epoch 5 --comment 'cvpr rebuttal lformer 2d reflash attn' \
# --ergas_ratio 4 \
# --pretrain --pretrain_id 'scy7mko6'


## lformer arch
  # python main.py --proj_name dcformer --arch mwsa_new \
  # -b 5 --device 'cuda:1' --dataset 'qb' --log_metrics --save_every_eval \
  # --warm_up_epochs 0 --num_worker 0 -e 2000 --aug_probs 0. 0. \
  # --loss 'l1ssim' --val_n_epoch 20 --logger_on --ergas_ratio 4 \
  # --comment 'mmnet on qb' \
  # --pretrain --pretrain_id '3dghlzjy'

## fuseformer arch
#  python main.py --proj_name dcformer --arch fuseformer\
#  -b 10 --device 'cuda:12' --dataset 'hisi' --logger_on \
#  --warm_up_epochs 30 --num_worker 0 -e 700 --aug_probs 0. 0. \
#  --loss l1 --val_n_epoch 10 --comment '训练fuseformer harvard' \
#  --pretrain --pretrain_id ufsb66w3

## hypert arch
#  python main.py --proj_name dcformer --arch hypertransformer \
#  -b 64 --device 'cuda:12' --dataset 'hisi' \
#  --warm_up_epochs 30 --num_worker 0 -e 500 --aug_probs 0. 0. \
#  --logger_on \
#  --loss l1 --val_n_epoch 20 --comment 'train hypertransformer on cave dataset' \
#  --pretrain --pretrain_id 2cmv1pb5

## Resume training script
#  python main.py --proj_name dcformer --arch dcformer --sub_arch 'reduce' \
#  -b 16 --device 'cuda:12' --dataset 'wv3' \
#  --warm_up_epochs 10 --num_worker 6 -e 600 --aug_probs 0. 0. \
#  --ddp --logger_on --load --resume_lr 1e-4 --resume_total_epochs 600 \
#  --loss mse --val_n_epoch 20 \
#  --comment 'dcformer_reduce on wv3 with less depth [4, [4, 3], [4, 3, 2], resume training' \
#  --run_id "lvu3ts9m"

#python main.py --proj_name panformer --arch panformer --sub_arch sga \
#  -b 8 --device 'cuda:12' \
#  --warm_up_epochs 10 --num_worker 6 -e 200 \
#  --ddp --wandb_on \
#  --loss mse --val_n_epoch 15 --comment 'test tensorboard logger'
  #--wandb_on \

# fusionnet
#python main.py --proj_name panformer --arch fusionnet \
#  -b 512 --ddp --num_worker 6 \
#  --resume 'None' --epochs 800 --loss mse --wandb_on


## test patch_merge_module
#python model/dcformer_reduce.py

