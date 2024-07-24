from .utils import (
    vis_saliency,
    vis_saliency_kde,
    click_select_position,
    grad_abs_norm,
    grad_norm,
    prepare_images,
    make_pil_grid,
    blend_input,
)
from .utils import cv2_to_pil, pil_to_cv2, gini, PIL2Tensor, Tensor2PIL
from .attributes import attr_grad
from .BackProp import I_gradient, attribution_objective, Path_gradient
from .BackProp import saliency_map_PG as saliency_map
from .BackProp import GaussianBlurPath, LinearPath
from .utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel

import numpy as np
# from beartype import beartype
import torch
import torch.nn as nn
import cv2

# @beartype
def LAM(
    model,
    hr: np.ndarray,
    lr_list: list[np.ndarray],
    h: int,
    w: int,
    sigma: float = 1.2,
    fold: int = 50,
    l: int = 9,
    alpha: float = 0.5,
    window_size: int = 16,
    *,
    on_cuda: bool = True
):

    if isinstance(model, nn.Module):
        assert (
            next(model.parameters()).is_cuda == on_cuda
        ), "parameters should {} be on cuda".format("on" if on_cuda else "not")
    else:
        assert callable(model), 'if @model is not an nn.Module, it should be a eval function of the model(a method)'

    tensor_hr = torch.tensor(hr)
    lr = lr_list[1][0] #ms
    tensor_lr = lr # torch.tensor(lr.numpy().cpu())

    img_hr = Tensor2PIL(tensor_hr[[0, 2, 4]])
    img_lr = Tensor2PIL(tensor_lr[[0, 2, 4]])
    
    draw_img = pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
    position_pil = cv2_to_pil(draw_img)

    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l) #LinearPath(fold)#
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(
        lr_list, model, attr_objective, gaus_blur_path_func, cuda=on_cuda
    )
    grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    print("gradient estimation done")
    
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=1)
    print('absolute saliency map done')
    # saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
    # print('kde saliency map done')
    
    
    blend_abs_and_input = cv2_to_pil(
        pil_to_cv2(saliency_image_abs) * (1.0 - alpha)
        + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha
    )
    # blend_kde_and_input = cv2_to_pil(
    #     pil_to_cv2(saliency_image_kde) * (1.0 - alpha)
    #     + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha
    # )
    pil = make_pil_grid(
        [
            position_pil,
            saliency_image_abs,
            blend_abs_and_input,
            # blend_kde_and_input,
            Tensor2PIL(torch.clamp(torch.from_numpy(result[:,[0,2,4]]), min=0.0, max=1.0)),
        ]
    )
    pil.save('result.png')
    
    return pil

