# Fourier-enhanced Implicit Neural Fusion Network for Multispectral and Hyperspectral Image Fusion

<div align="center">
<p style="text-align: center">
     <a href="https://scholar.google.com/citations?user=E5KO9XsAAAAJ&hl=en", style="font-size: 18px;">Yu-Jie Liang</a>
     &nbsp
     <a href="https://scholar.google.com/citations?user=pv61p_EAAAAJ&hl=en", style="font-size: 18px;">Zihan Cao</a>
     &nbsp
     <a href="https://scholar.google.com/citations?user=JZag1WIAAAAJ&hl=en", style="font-size: 18px;"> Shangqi Deng </a>
     &nbsp
     <a style="font-size: 18px;"> Hong-Xia Dou </a>
     &nbsp
     <a href="https://liangjiandeng.github.io/", style="font-size: 18px;"> Liang-Jian Deng </a>
     <br>
     <a style="font-size: 16px;"> University of Electronic Science & Technology of China </a>
</p>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2404.09293-b31b1b.svg)](https://arxiv.org/abs/2404.15174)

abtract: Recently, implicit neural representations (INR) have made significant strides in various vision-related domains, providing a novel solution for Multispectral and Hyperspectral Image Fusion (MHIF) tasks. However, INR is prone to losing high-frequency information and is confined to the lack of global perceptual capabilities. To address these issues, this paper introduces a Fourier-enhanced Implicit Neural Fusion Network (FeINFN) specifically designed for MHIF task, targeting the following phenomena: \textit{The Fourier amplitudes of the HR-HSI latent code and LR-HSI are remarkably similar; however, their phases exhibit different patterns.} In FeINFN, we innovatively propose a spatial and frequency implicit fusion function (Spa-Fre IFF), helping INR capture high-frequency information and expanding the receptive field. Besides, a new decoder employing a complex Gabor wavelet activation function, called Spatial-Frequency Interactive Decoder (SFID), is invented to enhance the interaction of INR features. Especially, we further theoretically prove that the Gabor wavelet activation possesses a time-frequency tightness property that favors learning the optimal bandwidths in the decoder. Experiments on two benchmark MHIF datasets verify the state-of-the-art (SOTA) performance of the proposed method, both visually and quantitatively. Also, ablation studies demonstrate the mentioned contributions.
<html>
<body>
    <div class="image-container" style="text-align: center;">
        <img src="figs/feinfn-teaser.png" alt="Image 1" width="100%">
    </div>
</body>
</html>

# Model
We implement FeINFN with Pytorch and you can find it at [`model/FeINFN.py`](../model/FeINFN.py).

## Traning
To train the model, running the following commands:

```shell
CUDA_VISIBLE_DEVICES="0" \
NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
OMP_NUM_THREADS="6" \
accelerate launch \
--config_file configs/huggingface/accelerate.yaml \
accelerate_main.py \
--proj_name FeINFN \
--arch FeINFN \
--dataset <dataset_name> \
--num_worker 0 -e 2000 -b 4 --aug_probs 0. 0. --loss l1ssim --grad_accum_steps 2 \
--checkpoint_every_n 20 --val_n_epoch 20  \
--comment "FeINFN config on <dataset_name> dataset model" \
--log_metric \
--logger_on \
```

> check the `model/__init__.py` if the FeINFN network is not registered.

Checkpoints, running, and Tensorboard logs will be saved at `log_file/`.

## Testing

You can refer to the testing script [`torch_inference_on_sharpening.py`](../torch_inference_on_sharpening.py) to test the model.

To test the metrics, please see the main guidance in [`README.md`](../README.md).

For sharpening tasks (including pansharpening and HMIF tasks), you simply test the metrics in Matlab:
``` matlab
cd Pansharpening_Hyper_SR_Matlab_Test_Package

%% when testing the reduced-resolution metrics on MHIF tasks
% Args:
% path: the saved fused image `.mat` file, find it in `visualized_img/`
% ratio: upscale ratio, e.g., 4
% full_res: we keep it to 0, not changed
% const: max value of the dataset (CAVE x4: 1, Harvard x4: 1, CAVE x8: 1, Harvard x8: 1)
analysis_ref_batched_images(path, ratio, full_res, const)
```

# Citation
If you find this work useful, please consider citing:
```bibtex
@article{liang2024fourier,
  title={Fourier-enhanced Implicit Neural Fusion Network for Multispectral and Hyperspectral Image Fusion},
  author={Liang, Yu-Jie and Cao, Zihan and Deng, Liang-Jian and Wu, Xiao},
  journal={arXiv preprint arXiv:2404.15174},
  year={2024}
}
```
