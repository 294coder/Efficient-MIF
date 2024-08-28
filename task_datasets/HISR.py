import time
from typing import Union
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision.transforms as T
import h5py
import random
from typing import List, Tuple, Optional, Callable

def default_dataset_fn(*x):
    return x[0]


class HISRDatasets(data.Dataset):
    # FIXME: when use this Dataset, you should set num_works to 0 or it will raise unpickable error
    def __init__(
        self,
        file: Union[h5py.File, str, dict],
        aug_prob=0.0,
        rgb_to_bgr=False,
        full_res=False,
        *,
        dataset_fn=None
    ):
        super(HISRDatasets, self).__init__()
        # warning: you should not save file (h5py.File) in this class,
        # or it will raise CAN NOT BE PICKLED error in multiprocessing
        # FIXME: should pass @path rather than @file which is h5py.File object to avoid can not be pickled error
        if isinstance(file, (str, h5py.File)):
            if isinstance(file, str):
                file = h5py.File(file)
            print(
                "warning: when @file is a h5py.File object, it can not be pickled.",
                "try to set DataLoader number_worker to 0",
            )
            
        # checking dataset_fn type
        if dataset_fn is not None:
            if isinstance(dataset_fn, (list, tuple)):
                def _apply_fn(tensor):
                    for fn in dataset_fn:
                        tensor = fn(tensor)
                    return tensor
                self.dataset_fn = _apply_fn 
            elif isinstance(dataset_fn, Callable):
                self.dataset_fn = dataset_fn
            else: raise TypeError("dataset_fn should be a list of callable or a callable object")
        else:
            self.dataset_fn = default_dataset_fn
        
        self.full_res = full_res
        data_s= self._split_parts(
            file, rgb_to_bgr=rgb_to_bgr, full=full_res
        )
        
        if len(data_s) == 4:
            self.gt, self.lr_hsi, self.rgb, self.hsi_up = data_s

        else:
            self.lr_hsi, self.rgb, self.hsi_up = data_s           
        
        self.size = self.rgb.shape[-2:]
        print("dataset shape:")

        # print dataset info
        if not full_res:
            print("{:^20}{:^20}{:^20}{:^20}".format("lr_hsi", "hsi_up", "rgb", "gt"))
            print(
                "{:^20}{:^20}{:^20}{:^20}".format(
                    str(tuple(self.lr_hsi.shape)),
                    str(tuple(self.hsi_up.shape)),
                    str(tuple(self.rgb.shape)),
                    str(tuple(self.gt.shape)),
                )
            )
            
        else:
            print("{:^20}{:^20}{:^20}".format("lr_hsi", "hsi_up", "rgb"))
            print(
                "{:^20}{:^20}{:^20}".format(
                    str(tuple(self.lr_hsi.shape)),
                    str(tuple(self.hsi_up.shape)),
                    str(tuple(self.rgb.shape)),
                )
            )
            
        # geometrical transformation
        self.aug_prob = aug_prob
        self.geo_trans = (
            T.Compose(
                [
                    # T.RandomHorizontalFlip(p=self.aug_prob),
                    # T.RandomVerticalFlip(p=self.aug_prob),
                    T.RandomApply(
                        [
                            T.RandomErasing(
                                p=self.aug_prob, scale=(0.02, 0.15), ratio=(0.2, 1.0)
                            ),
                            T.RandomAffine(
                                degrees=(0, 70),
                                translate=(0.1, 0.2),
                                scale=(0.95, 1.2),
                                interpolation=T.InterpolationMode.BILINEAR,
                            ),
                        ],
                        p=self.aug_prob,
                    ),
                    # T.RandomAutocontrast(p=self.aug_prob),
                    # T.RandomAdjustSharpness(sharpness_factor=2, p=self.aug_prob)
                    # T.RandomErasing(p=self.aug_prob)
                ]
            )
            if aug_prob != 0.0
            else lambda *x: x
        )

    def _split_parts(self, file, load_all=True, rgb_to_bgr=False, keys=None, full=False):
        # has already been normalized
        
        
        # warning: key RGB is HRMSI when the dataset is GF5-GF1
        if not full:
            keys = ['GT', 'LRHSI', 'RGB', 'HSI_up']
        else:
            keys = ['LRHSI', 'RGB', 'HSI_up']
        
        if load_all:
            # load all data in memory
            data = []
            for k in keys:
                data.append(
                    self.dataset_fn(torch.tensor(file[k][:], dtype=torch.float32)),
                )
            if rgb_to_bgr:
                print("warning: rgb to bgr, for testing generalization only.")
                # rgb -> bgr
                if not full:
                    data[2] = data[2][:, [-1, 1, 0]]
                else:
                    data[1] = data[1][:, [-1, 1, 0]]
            return data
        else:
            # warning: it will ignore @normalize
            # warning: "GT" can not be access in FULL mode
            return (
                file.get("GT"),
                file.get("LRHSI"),
                file.get("RGB"),
                file.get("HSI_up"),
            )

    def aug_trans(self, *data):
        data_list = []
        seed = torch.random.seed()
        for d in data:
            torch.manual_seed(seed)
            random.seed(seed)
            d = self.geo_trans(d)
            data_list.append(d)
        return tuple(data_list)

    def __getitem__(self, index):
        # gt: [31, 64, 64]
        # lr_hsi: [31, 16, 16]
        # rbg: [3, 64, 64]
        # hsi_up: [31, 64, 64]

        # harvard [rgb]
        # cave [bgr]
        if not self.full_res:
            tuple_data = (
                self.rgb[index],
                self.lr_hsi[index],
                self.hsi_up[index],
                self.gt[index],
            )
        else:
            tuple_data = (
                self.rgb[index],
                self.lr_hsi[index],
                self.hsi_up[index],
            )
        if self.aug_prob != 0.0:
            return self.aug_trans(*tuple_data)
        else:
            return tuple_data

    def __len__(self):
        return len(self.rgb)
