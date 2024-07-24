## panRWKV_v3
NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
OMP_NUM_THREADS="6" \
python -u -m accelerate.commands.launch \
--gpu_ids "1" accelerate_inference_on_fusion.py \
--arch panRWKV --sub-arch v3 \
--val-bs 1 --dataset msrs \
--dataset-mode 'detection' \
--extra-save-name joint_only_y \
--model-path "/Data3/cao/ZiHanCao/exps/panformer/log_file/panRWKV_v3/vis_ir_joint/2024-07-23-01-01-54_panRWKV_n5d4qwzu_panRWKV config on msrs dataset tiny model/weights/ema_model.pth" \
--only-y
# --debug

## MGDN
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# python -u -m accelerate.commands.launch \
# --gpu_ids "0" accelerate_inference_on_fusion.py \
# --arch panRWKV --sub-arch v3 \
# --val-bs 1 --dataset tno \
# --model-path "/Data3/cao/ZiHanCao/exps/panformer/weight/2024-06-25-17-06-20_panRWKV_mezc5fqb/panRWKV_mezc5fqb.pth" \
# --extra-save-name 'direct_train_v2'
