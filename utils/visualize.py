import os
import os.path as osp
from typing import Literal, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from utils.misc import to_numpy

def get_rgb_channel_by_dataset_name(tensor, dataset_name: str):
    if dataset_name in ('wv3', 'wv2'):
        return tensor[:, [4,2,0], ...]
    elif dataset_name in ('gf2', 'qb'):
        return tensor[:, :3, ...]
    elif dataset_name in ('gf5', 'gf5-gf1'):
        return tensor[:, [40, 30, 20], ...]
    elif dataset_name == 'houston':
        return tensor[:, [39, 29, 19], ...]
    elif 'cave' in dataset_name or 'harvard' in dataset_name:
        return tensor[:, [29, 19, 9], ...]
    else:
        return tensor[:, :3, ...]


def permute_dim(*args):
    d = [
        i.permute(2, 0, 1) for i in args
    ]
    return d


def normalize(img, to_uint8=True):
    """
    centering image to show
    :param img: numpy array, shape [H, W, C]
    :return: uint8 type image
    """
    img = img - img.min((0, 1))
    img = img / img.max((0, 1))
    if to_uint8:
        img *= 255
        img = img.astype('uint8')
    return img


def invert_normalized(norm_img: Union[Tensor, np.ndarray],
                      mean: Union[Tensor, np.ndarray],
                      std: Union[Tensor, np.ndarray],
                      *,
                      change_back_dim=True):
    """
    invert image normalized to unnormalized image
    Args:
        norm_img: Tensor: [C, H, W] or [B, C, H, W]
        mean: Tensor, [C, ]
        std: Tensor, [C, ]
        change_back_dim: bool, change channel dim back

    Returns: Ndarray, unnormalized image

    """
    if isinstance(norm_img, Tensor):
        norm_img = to_numpy(norm_img)[0]
    if isinstance(mean, Tensor):
        mean = to_numpy(mean)[0]
    if isinstance(std, Tensor):
        std = to_numpy(std)[0]
    if norm_img.ndim == 4:
        _dim_trans = [0, 2, 3, 1]
        _dim_back_trans = [0, -1, 1, 2]
    else:
        _dim_trans = [1, 2, 0]
        _dim_back_trans = [2, 0, 1]
    norm_img = norm_img.transpose(_dim_trans)

    assert norm_img.shape[-1] == mean.size == std.size
    unnormed_img = norm_img * std + mean  # [H, W, C] or [B, H, W, C]
    if change_back_dim:
        unnormed_img = unnormed_img.transpose(_dim_back_trans)
    return unnormed_img


def hist_equal(img):
    """
    equalize an image
    :param img: numpy array, shape [H, W, C], C can be any int
    :return:
    """
    if img.ndim == 3:
        for i in range(img.shape[-1]):
            img[..., i] = cv2.equalizeHist(img[..., i])
    else:
        img = cv2.equalizeHist(img)
    return img


def res_image(gt: Tensor, sr: Tensor, *, exaggerate_ratio: int = None) -> torch.Tensor:
    # shape [B, C, H, W]
    ratio = exaggerate_ratio if exaggerate_ratio is not None else 1.
    res = torch.abs(gt - sr).mean(1, keepdim=True) * ratio
    return res


def get_spectral_image_ready(batch_image: Tensor, 
                             tensor_name: str, 
                             task: str=None,
                             ds_name: Literal['wv3', 'wv2', 'gf2', 'qb',
                                              'gf5', 'gf5-gf1', 'flir', 'tno',
                                              'msrs']=None) -> Tensor:
    # batch_image: [B, C, H, W]
    img_arrs = batch_image.permute(0, 2, 3, 1).detach().cpu().numpy()  # [B, H, W, C]
    if task == 'fusion': 
        transform_fn = lambda x, const: torch.tensor(x, dtype=torch.float32)
    elif task == 'sharpening':
        if tensor_name in ('lms', 'ms', 'sr'):  # for pansharpening and HISR tasks
            batch_image = get_rgb_channel_by_dataset_name(batch_image, ds_name)
        elif tensor_name == 'pan' and batch_image.shape[1] > 3:
            batch_image = batch_image[:, :3]
        
        transform_fn = lambda x, const: torch.tensor(normalize(x, to_uint8=False) * const)
    else:
        raise ValueError(f'Invalid task: {task}')
        
    if 'res' in tensor_name:
        equalized_img = [transform_fn(i, 10).permute(-1, 0, 1)[None, ...] for i in img_arrs]  # [1, C, H, W]
    else:
        equalized_img = [transform_fn(i, 1).permute(-1, 0, 1)[None, ...] for i in img_arrs]
        
    grid = torch.cat(equalized_img, dim=0)
    
    return grid


def viz_batch(img: Tensor, base_path='./visualized_img', suffix=None, start_index=1, format='jpg'):
    assert suffix is not None, 'arg @suffix can not be None'
    assert suffix in ['pan', 'ms', 'sr', 'gt', 'residual'], 'arg @suffix should only be pan, ms or sr'
    img_arrs = img.permute(0, 2, 3, 1).numpy()
    if suffix == 'residual':
        equalized_img = [i for i in img_arrs]
    else:
        # equalized_img = [hist_equal(normalize(i)) for i in img_arrs]
        equalized_img = [normalize(i) for i in img_arrs]

    path = osp.join(base_path, suffix)
    if not osp.exists(path):
        os.makedirs(path)
    # all_path = path + '.mat'
    # savemat(all_path, {f'{suffix}': img_arrs})

    # fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

    # ax: plt.Axes
    # fig: plt.Figure
    for i, img in enumerate(equalized_img, start_index):
        h, w = img.shape[-2:]
        img_path1 = osp.join(path, str(i) + '.' + format)
        # plt.cla()
        if suffix == 'pan' or suffix == 'residual':
            # ax.imshow(img, cmap='gray')
            cv2.imwrite(img_path1, img)
        # elif suffix == 'residual':
        #     ax.imshow(img)
        else:
            try:
                # ax.imshow(img[..., [0, 2, 4]])
                cv2.imwrite(img_path1, img[..., [0, 2, 4]])
            except:
                # ax.imshow(img[..., :3])
                cv2.imwrite(img_path1, img[..., :3])
    #     ax.set_axis_off()
    #     fig.set_size_inches(h, w)
    #     fig.savefig(img_path1, format=format, dpi=50, bbox_inches='tight', pad_inches=0.)
    # plt.close()


def show_details(img: np.ndarray,
                 cpos_ratio: Tuple[float, float],
                 area_pixels: Tuple[int, int],
                 interp_ratio: int = 3,
                 color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2,
                 place: str = None) -> np.ndarray:
    """select a patch in raw image which decided by @cpos_ratio and @area_pixels,
    the function will interpolate @interp_ratio times and paste it in a corner.

    Args:
        img (np.ndarray): raw image needed to detailed, format [H, W, C]
        cpos_ratio (Tuple[float, float]): selected patch's centroid, from 0 to 1
        area_pixels (Tuple[int, int]): pixel area of the patch
        interp_ratio (int, optional): interpolate ratio. Defaults to 3.
        color (Tuple[int, int, int], optional): color of the box. Defaults to (0, 255, 0).
                                                recommend colors:
                                                    (236,229,240)
                                                    (233,138,21)
                                                    (0,59,54)
        thickness (int, optional): thickness of the box. Defaults to 2.
        place(str, optional): where to place the interpolated patch.
    """
    assert 0 < cpos_ratio[0] < 1 and 0 < cpos_ratio[1] < 1, '@cpos_ratio can only be range (0, 1)'

    # to array
    if img.ndim == 2:
        img = np.repeat(img[..., np.newaxis], 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    img_size = np.array(img.shape[:2])
    cpos_pixels = np.array(cpos_ratio) * img_size
    cpos_ratio = np.array(cpos_ratio)
    area_pixels = np.array(area_pixels)
    paste_pixels = area_pixels * interp_ratio
    img = img.astype('uint8')
    img = np.ascontiguousarray(img)

    # bound check
    bound = []
    for i, j in zip((-1, 1), (-1, 1)):
        bound.append([i * area_pixels[0] / 2, j * area_pixels[1] / 2])
    bound = np.array(bound)
    bound = np.repeat(cpos_pixels[np.newaxis, ...], 2, axis=0) + bound
    assert not np.bitwise_or(bound[0] < 0, bound[1] > img_size).any(), \
        f'selected range out of image size, image size {img_size} but get selected range {bound}'

    # find furthest corner to paste the interpolated patch
    if place is None:
        furthest_pos_ratio = None
        furthest_dis = 0.
        for i in (0, 1):
            for j in (0, 1):
                d = (i - cpos_ratio[0]) ** 2 + (j - cpos_ratio[1]) ** 2
                if d > furthest_dis:
                    furthest_pos_ratio = (i, j)
                    furthest_dis = d
    else:
        assert place in ('lt', 'rt', 'lb', 'rb'), '@place should be one of [lt, rt, lb, rb]'
        place_dict = {'lt': (0, 0), 'rt': (0, 1), 'lb': (1, 0), 'rb': (1, 1)}
        furthest_pos_ratio = place_dict[place]

    bound = bound.astype('int')
    patch = img[bound[0, 0]:bound[1, 0], bound[0, 1]: bound[1, 1], :]
    interp_img = cv2.resize(patch, dsize=paste_pixels[::-1])

    box_edge_point = []
    cv2.rectangle(img, bound[0][::-1], bound[1][::-1], color, thickness=thickness)
    box_pre_thick = thickness // 2
    if furthest_pos_ratio == (0, 0):
        img[:paste_pixels[0], :paste_pixels[1], :] = interp_img
        box_edge_point = [[box_pre_thick, box_pre_thick], paste_pixels[::-1]]
    elif furthest_pos_ratio == (1, 0):
        img[-paste_pixels[0]:, :paste_pixels[1], :] = interp_img
        box_edge_point = [[box_pre_thick, img_size[0] - paste_pixels[0]],
                          [paste_pixels[1], img_size[0] - box_pre_thick]]
    elif furthest_pos_ratio == (0, 1):
        img[:paste_pixels[0], -paste_pixels[1]:, :] = interp_img
        box_edge_point = [[img_size[1] - paste_pixels[1], box_pre_thick],
                          [img_size[1] - box_pre_thick, paste_pixels[0]]]
    else:
        img[-paste_pixels[0]:, -paste_pixels[1]:, :] = interp_img
        box_edge_point = [[img_size[1] - paste_pixels[1], img_size[0] - paste_pixels[0]],
                          [img_size[1] - box_pre_thick, img_size[0] - box_pre_thick]]
    cv2.rectangle(img, box_edge_point[0], box_edge_point[1], color, thickness)

    return img


def plt_plot_img_without_white_margin(img, *args, **kwargs):
    """

    :param img: format [H, W, C]
    :param args: plt.imshow args
    :param kwargs: plt.imshow kwargs
    :return:
    """
    width, height = img.shape[:2]

    ax = plt.imshow(img, *args, **kwargs)

    fig = plt.gcf()
    fig.set_size_inches(width / 100, height / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().set_axis_off()

    return fig, ax

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import PIL.Image as Image

    img = Image.open('../visualized_img/sr/8.eps')
    img = np.asarray(img)
    img = show_details(img, cpos_ratio=(0.2, 0.8), area_pixels=(50, 50), thickness=2)

    plt_plot_img_without_white_margin(img)
    plt.show()
