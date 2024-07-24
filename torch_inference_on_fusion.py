import h5py
import numpy as np
import torch as th

from task_datasets.RoadScene import RoadSceneDataset
from task_datasets.TNO import TNODataset, WindowBasedPadder
from utils import AnalysisVISIRAcc

# path = "/media/office-401-remote/Elements SE/cao/ZiHanCao/datasets/RoadSceneFusion_1"
path = '/media/office-401-remote/Elements SE/cao/ZiHanCao/datasets/TNO'
# ds = FLIRDataset(path, "test", no_split=True)
ds = TNODataset(path, "test", no_split=True)
dl = th.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

import torch

from model.build_network import build_network

device = torch.device("cuda:0")
torch.cuda.set_device(device)

# net = build_network(
#     "dcformer_mwsa",
#     spectral_num=1,
#     added_c=1,
#     block_list=[1, [1, 1], [1, 1, 1]],
#     mode="C",
#     residual=False,
# )
net = build_network(
    'dcformer_mwsa',
    spectral_num=1,
    block_list=[1, [1, 1], [1, 1, 1]],
    mode="C",
    added_c=1
)
net.load_state_dict(
    # th.load('/home/ZiHanCao/exps/panformer/weight/dcformer_379zkf3e/ep_550.pth', map_location=device)['model']  # 2n8eo45b
    # th.load('./weight/dcformer_17rgbfmz/ep_490.pth', map_location=device)['model']
    # th.load("./weight/dcformer_1qxd4k6t/ep_620.pth", map_location=device)["model"]  # mgimci loss
    # th.load("./weight/dcformer_1at6je5u/ep_1280.pth", map_location=device)["model"]  # pia loss
    # th.load("./weight/dcformer_10yvuovr/ep_1300.pth", map_location=device)["model"]  # pia loss + lpips loss
    # th.load("./weight/dcformer_10m1crxr/ep_920.pth", map_location=device)['model']  # pia loss + lpips loss and TNO new training dataset
    # th.load("./weight/dcformer_3m7v12ou/ep_200.pth", map_location=device)['model']  # TNO and RoadScence dataset toghter training
    # th.load("./weight/dcformer_2705u7eg/ep_1160.pth", map_location=device)['model']  # u2fusion original loss, not work, patch artifact
    # th.load("./weight/dcformer_332f31gf/ep_720.pth", map_location=device)["model"]
    # th.load("./weight/dcformer_13g4q0ls/ep_440.pth", map_location=device)["model"]
    # th.load("./weight/dcformer_34jj20so/ep_60.pth", map_location=device)["model"]
    # th.load("./weight/dcformer_1beukp0d.pth", map_location=device)["model"]
    # th.load("./weight/dcformer_3umay70j/ep_260.pth", map_location=device)["model"]  # cddloss
    # th.load("./weight/dcformer_709y225a/ep_120.pth", map_location=device)["model"]
    
    # th.load('./weight/dcformer_k81srmy1/ep_120.pth', map_location=device)['model']  # u2fusion loss, good, TNO
    # th.load('./weight/dcformer_23tbaql6/ep_230.pth', map_location=device)['model']  # u2fusion loss, RoadScene (and TNO), test only RS
    
    
    
    # ydtr
    # th.load('./weight/ydtr.pth', map_location=device)# ['model']
)
base_ws = 16
net = net.cuda()

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.io import savemat
from torchvision.utils import make_grid
from tqdm import tqdm

from model.base_model import PatchMergeModule


def convert_uint8(img):
    print(img.shape, end=" ")
    if img.dtype != np.uint8:
        img = img.clip(0, 1)
        img *= 255
        # print('convert to [0, 255]')
    return img.astype(np.uint8)


net.eval()
# patch_merge_net = PatchMergeModule(net, 1, patch_size=128, scale=1, device=device)
padder = WindowBasedPadder(64)
ms_padder = WindowBasedPadder(64)
acc_analysiser = AnalysisVISIRAcc()
with th.no_grad():
    for i, (ir, ms, vis, gt) in tqdm(enumerate(dl), total=dl.dataset.__len__()):
        ir, ms, vis, gt = ir.cuda(), ms.cuda(), vis.cuda(), gt.cuda()

        ir = padder(ir)
        vis = padder(vis, no_check_pad=True)
        
        shape = ir.shape[-2:]
        ms_shape = (torch.tensor(shape) // 4).tolist()
        ms = ms_padder(ms, size=ms_shape)
        
        # spa_size = gt.shape[-2:]
        # pan_nc = ir.size(1)
        # ms_nc = ms.size(1)
        # input = (
        #     F.interpolate(
        #         ms, size=tuple(vis.shape[-2:]), mode="bilinear", align_corners=True
        #     ),
        #     vis,
        #     torch.cat([ir, torch.zeros(1, ms_nc - pan_nc, *spa_size).cuda()], dim=1),
        # )
        # sr = patch_merge_net.forward_chop(*input)[0]
        
        max_k = ir.shape[-1]
        
        window_dict = {}
        for j in range(3):
            window_dict[max_k // (2**j)] = base_ws // (2**j)
        
        net._set_window_dict(window_dict)
        sr = net.val_step(ms, vis, ir)
        sr = padder.inverse(sr)
        
        acc_analysiser(gt, sr)
        
        sr = sr.detach().cpu().numpy()[0]
        sr_show = sr.transpose([1, 2, 0])
        vis_show = vis.detach().cpu().numpy()[0].transpose([1, 2, 0])
        ir_show = ir.detach().cpu().numpy()[0].transpose([1, 2, 0])

        fig, axes = plt.subplots(ncols=3, figsize=(12, 4), dpi=200)
        axes = axes.flatten()

        for img, name, ax in zip(
                [vis_show, ir_show, sr_show], ["vis", "ir", "fuse"], axes
        ):
            ax.imshow(img, "gray")
            ax.set_axis_off()
            ax.set_title(name)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(pad=0)
        # plt.show()

        sr_show = convert_uint8(sr_show)
        cv2.imwrite(f"./visualized_img/ir/{i}.bmp", sr_show)
        print("img saved to {}".format(f"./visualized_img/ir/{i}.bmp"))
        
    print(acc_analysiser.result_str())
