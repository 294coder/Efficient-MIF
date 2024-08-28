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
# CUDA_VISIBLE_DEVICES="0" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# python -u -m accelerate.commands.launch \
# --config_file configs/huggingface/accelerate.yaml \
# accelerate_run_main.py \
# --proj_name panRWKV_v3 \
# --arch panRWKV \
# --sub_arch v3 \
# --dataset wv3 \
# --num_worker 0 -e 800 --train_bs 64 --val_bs 1 \
# --aug_probs 0. 0. --loss l1ssim --grad_accum_steps 1 \
# --val_n_epoch 10 \
# --comment "panRWKV config on WV3 dataset tiny model without q_shift" \
# --checkpoint_every_n 10 \
# --metric_name_for_save "SAM" \
# --log_metric --logger_on \
# --pretrain_model_path "/Data3/cao/ZiHanCao/exps/panformer/weight/2024-06-10-20-12-19_panRWKV_z9ydu64u/panRWKV_z9ydu64u.pth" \
# --non_load_strict
# --resume_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/panRWKV_v3/gf2/2024-07-28-21-59-11_panRWKV_nsruqx06_panRWKV config on GF2 dataset tiny model/weights/ep_10"
# --sanity_check \


# panRWKV_v3 MEF
# CUDA_VISIBLE_DEVICES="0" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# python -u -m accelerate.commands.launch \
# --config_file configs/huggingface/accelerate.yaml \
# accelerate_run_main.py \
# --proj_name panRWKV_v3 \
# --arch panRWKV \
# --sub_arch v3 \
# --dataset vis_ir_joint \
# --num_worker 0 -e 800 --train_bs 18 --val_bs 1 \
# --aug_probs 0.0 0. --loss drmffusion --grad_accum_steps 2 \
# --val_n_epoch -1 \
# --comment "panRWKV config on vis_ir_joint dataset tiny model" \
# --checkpoint_every_n 10 \
# --metric_name_for_save "PSNR" \
# --fusion_crop_size 96 \
# --log_metric --logger_on \
# --pretrain_model_path '/Data3/cao/ZiHanCao/exps/panformer/log_file/panRWKV_v3/vis_ir_joint/2024-07-28-15-39-39_panRWKV_v64628cj_panRWKV config on vis_ir_joint dataset tiny model/weights/ep_10/model.safetensors' \


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


## panRWKV_v4 pansharpening and HMIF
# CUDA_VISIBLE_DEVICES="0,1,2" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# python -u -m accelerate.commands.launch \
# --config_file configs/huggingface/accelerate.yaml \
# --multi_gpu \
# --num_processes 3 \
# --gpu_ids "0,1,2" \
# accelerate_run_main.py \
# --proj_name panRWKV_v8_cond_norm \
# --arch panRWKV \
# --sub_arch v8_cond_norm \
# --dataset qb \
# --num_worker 0 -e 800 --train_bs 48 --val_bs 1 \
# --aug_probs 0. 0. --loss l1ssim --grad_accum_steps 1 \
# --val_n_epoch 10 \
# --comment "k=2 with q shift conditional scale shift gated for attn and ffn with parallel MIFM" \
# --metric_name_for_save "SAM" \
# --log_metric --logger_on \
# --pretrain_model_path "/Data2/ZiHanCao/exps/panformer/log_file/panRWKV_v8_cond_norm/gf2/2024-08-09-03-09-26_panRWKV_u4o64esy_k=2 with q shift conditional scale shift gated for attn and ffn with parallel MIFM/weights/ema_model.pth" \
# --non_load_strict
# --checkpoint_every_n None\


## panRWKV_v8 VIS IR joint
CUDA_VISIBLE_DEVICES="0,1" \
NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
OMP_NUM_THREADS="6" \
python -u -m accelerate.commands.launch \
--config_file configs/huggingface/accelerate.yaml \
--num_processes 1 \
--gpu_ids "0" \
accelerate_run_main.py \
--proj_name panRWKV_v8_cond_norm \
-m 'panrwkv_v8_cond_norm.RWKVFusion' \
-c 'panRWKV_config.yaml' \
--dataset vis_ir_joint \
--num_worker 0 -e 600 --train_bs 40 --val_bs 1 \
--aug_probs 0. 0. --loss drmffusion --grad_accum_steps 2 \
--val_n_epoch 10 \
--comment "RWKVFusion_v8_cond_norm with mean prior fusion" \
--logger_on \
--fast_eval_n_samples 80 \
--checkpoint_every_n 10 \

# --pretrain_model_path "log_file/panRWKV_v8_cond_norm/vis_ir_joint/2024-08-20-14-33-19_panRWKV_5qx8inx8_PanRWKV_v8_cond_norm/weights/ema_model.pth"
# --regardless_metrics_save \
# --sanity_check
# --resume_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/panRWKV_v8_cond_norm/vis_ir_joint/2024-08-19-22-05-44_panRWKV_nuw9349k_PanRWKV_v8_cond_norm/weights/checkpoint_0" \
# --pretrain_model_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/panRWKV_v8_cond_norm/vis_ir_joint/2024-08-19-02-01-39_panRWKV_h5bat77f_k=2 with q shift conditional scale shift gated for attn and ffn with parallel MIFM/weights/ema_model2.pth" \
# --non_load_strict


## panRWKV_plain_v6
# CUDA_VISIBLE_DEVICES="0,1" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# python -u -m accelerate.commands.launch \
# --config_file configs/huggingface/accelerate.yaml \
# --num_processes 1 \
# --gpu_ids "0" \
# accelerate_run_main.py \
# --proj_name RWKVFusion_plain_v6 \
# -c panRWKV_config.yaml \
# -m panRWKV_plain_v6.RWKVFusion \
# --dataset wv3 \
# --num_worker 0 -e 600 --train_bs 20 --val_bs 1 \
# --aug_probs 0. 0. --loss l1ssim --grad_accum_steps 2 \
# --val_n_epoch 10 \
# --comment "RWKVFusion_plain_v6" \
# --logger_on \

# --fast_eval_n_samples 80 \