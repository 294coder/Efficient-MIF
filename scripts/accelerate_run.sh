## accelerate run
# CUDA_VISIBLE_DEVICES="0" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch --config_file configs/huggingface/accelerate.yaml accelerate_main.py \
# --proj_name panRWKV_v3 --arch panRWKV --sub_arch v3 --dataset wv3 \
# --num_worker 6 -e 800 -b 32 --aug_probs 0. 0. --loss l1ssim --grad_accum_steps 2 \
# --checkpoint_every_n 20 --val_n_epoch 20  \
# --comment "panRWKV config on wv3 dataset model" \
# --log_metric \
# --logger_on \
# --resume_path "/Data2/ZiHanCao/exps/panformer/weight/2024-04-17-02-07-23_panRWKV_9sqz9900/ep_800"
# --pretrain_model_path "/Data2/ZiHanCao/exps/panformer/weight/2024-04-15-20-24-17_panRWKV_96z6zd29/panRWKV_96z6zd29.pth" \
# --non_load_strict

# NCCL_SOCKET_IFNAME="eth0" \


## accelerate test
# CUDA_VISIBLE_DEVICES="1,2" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# python -u -m accelerate.commands.launch --gpu_ids "0,1" \
# --multi_gpu --num_processes 2 accelerate_test.py

## panRWKV_v3
CUDA_VISIBLE_DEVICES="1" \
NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
OMP_NUM_THREADS="6" \
python -u -m accelerate.commands.launch \
--config_file configs/huggingface/accelerate.yaml \
accelerate_run_main.py \
--proj_name panRWKV_v3 \
--arch panRWKV \
--sub_arch v3 \
--dataset vis_ir_joint \
--num_worker 0 -e 800 --train_bs 18 --val_bs 1 \
--aug_probs 0.2 0. --loss drmffusion --grad_accum_steps 2 \
--val_n_epoch -1 \
--comment "panRWKV config on msrs dataset tiny model" \
--checkpoint_every_n 10 \
--metric_name_for_save "PSNR" \
--fusion_crop_size 96 \
--log_metric --logger_on \
# --only_y \
# --non_load_strict \
# --pretrain_model_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/panRWKV_v3/msrs/2024-07-22-11-10-04_panRWKV_3cjvjrfm_panRWKV config on msrs dataset tiny model/weights/ema_model.pth" \
# --resume_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/panRWKV_v3/vis_ir_joint/2024-07-17-23-31-47_panRWKV_utyljcny_panRWKV config on msrs dataset tiny model with segmentaion task/weights/ep_30" \
# --sanity_check \




## MGDN
# CUDA_VISIBLE_DEVICES="0" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --config_file configs/huggingface/accelerate.yaml \
# accelerate_run_main.py \
# --proj_name MGDN --arch MGDN --dataset med_harvard \
# --num_worker 0 -e 400 --train_bs 14 --val_bs 1 \
# --aug_probs 0. 0. --loss drmffusion --grad_accum_steps 1 \
# --val_n_epoch -1  \
# --comment "MGDN config on MSRS dataset base model" \
# --log_metric \
# --checkpoint_every_n 10 \
# --metric_name_for_save "PSNR" \
# --logger_on \
# --sanity_check
