## panRWKV_v3
NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
OMP_NUM_THREADS="6" \
python -u -m accelerate.commands.launch \
--gpu_ids "0" accelerate_inference_on_fusion.py \
-c "configs/panRWKV_config.yaml" \
-m "panrwkv_v8_cond_norm.RWKVFusion" \
--val-bs 1 --dataset msrs \
--dataset-mode 'test' \
--extra-save-name no_prior \
--model-path "log_file/panrwkv_v8_cond_norm.RWKVFusion/vis_ir_joint/2024-08-28-14-49-11_panRWKV_k30dlgmi_RWKVFusion_v8_cond_norm no prior fusion/weights/ema_model.pth" \
# --load-spec-key 'ema_model'

# --analysis-fused
# --only-y
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
