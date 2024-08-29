# Linearly-evolved Transformer for Pan-sharpening
<div align="center">
<p style="text-align: center">
     <a style="font-size: 18px;"> JunMing Hou* </a>
     &nbsp
     <a href="https://scholar.google.com/citations?user=pv61p_EAAAAJ&hl=en", style="font-size: 18px;">Zihan Cao*</a>
     &nbsp
     <a style="font-size: 18px;"> Naishan Zheng </a>
     &nbsp
     <a style="font-size: 18px;"> Xuan Li </a>
     &nbsp
     <a style="font-size: 18px;"> Xiaoyu Chen </a>
     &nbsp
     <br>
     <a style="font-size: 18px;"> XinYang Li </a>
     &nbsp
     <a style="font-size: 18px;"> Xiaofeng Cong</a>
     &nbsp
     <a style="font-size: 18px;"> Man Zhou </a>
     &nbsp
     <a style="font-size: 18px;"> Danfeng Hong </a>
     &nbsp
     <br>
     <a style="font-size: 16px;"> University of Electronic Science Technology of China </a>
     <br>
     <a style="font-size: 16px;"> Southeast University </a>
     <br>
     <a style="font-size: 16px;"> University of Science and Technology </a>
</p>
</div>


[![arXiv](https://img.shields.io/badge/arXiv-2404.12804-b31b1b.svg)](https://arxiv.org/abs/2404.12804)


# Fast testing

We provide [pretrained weights](链接：https://pan.baidu.com/s/1keK5eAIrZcPPgoEr8bcW5A?pwd=y2t9) and a fast testing script to test the performance of our model.

To run the testing script, please refer to `torch_inference_on_sharpening.py` and adapt following steps:
1. modify the `path` for datset;
2. change the `dataset_type`;
3. set `full_res` to the full resolution or reduced resolution of datasets (for pansharpening).
4. change the yaml file for configurate the model. For LFormer, it's in `configs/lformer_config.yaml`.

```yaml
network_configs:
  lformer:
    pan_dim: 1
    lms_dim: 4 # 4 for GF2, 8 for WV3, 31 for CAVE x4
    attn_dim: 64
    hp_dim: 64
    n_stage: 5
    patch_merge: yes
    crop_batch_size: 64
    patch_size_list: [16, 64, 64]
    scale: 4
```

# Train

You can train the LFormer model by run the commands:
```shell
CUDA_VISIBLE_DEVICES="0" \
NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
OMP_NUM_THREADS="6" \
accelerate launch \
--config_file configs/huggingface/accelerate.yaml \
accelerate_main.py \
--proj_name LFormer \
--arch LFormer \
--dataset <dataset_name> \
--num_worker 6 -e 800 -b 32 --aug_probs 0. 0. --loss l1ssim --grad_accum_steps 2 \
--checkpoint_every_n 20 --val_n_epoch 20  \
--comment "LFormer config on wv3 dataset model" \
--log_metric \
--logger_on \
```
> Other DDP training (or using deepspeed, please see Huggingface Accelerate documenation).

# Citation

If you find this code useful for your research, please consider citing:

```
@inproceedings{
hou2024linearlyevolved,
title={Linearly-evolved Transformer for Pan-sharpening},
author={Junming Hou and Zihan Cao and Naishan Zheng and Xuan Li and Xiaoyu Chen and Xinyang Liu and Xiaofeng Cong and Danfeng Hong and Man Zhou},
booktitle={ACM Multimedia 2024},
year={2024},
url={https://openreview.net/forum?id=pCxZTmGr4O}
}
```