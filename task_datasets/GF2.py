import torch
import torch.utils.data as data
import torchvision.transforms as T
import cv2
import numpy as np
import h5py
from typing import List, Tuple, Optional

class Identity:
    def __call__(self, *args):
        return args


class GF2Datasets(data.Dataset):
    def __init__(
        self,
        d,
        aug_prob=0.0,
        hp=False,
        hp_ksize=(5, 5),
        norm_range=True,
        full_res=False,
        const=1023.0,
    ):
        """

        :param d: h5py.File or dict or path
        :param aug_prob: augmentation probability
        :param hp: high pass for ms and pan. x = x - cv2.boxFilter(x)
        :param hp_ksize: cv2.boxFiler kernel size
        :param norm_range: normalize data range
        """
        super(GF2Datasets, self).__init__()
        # FIXME: should pass @path rather than @file which is h5py.File object to avoid can not be pickled error
        if isinstance(d, (str, h5py.File)):
            if isinstance(d, str):
                d = h5py.File(d)
            print(
                "warning: when @file is a h5py.File object, it can not be pickled.",
                "try to set DataLoader number_worker to 0",
            )
        if not full_res:
            self.gt, self.ms, self.lms, self.pan = self.get_divided(d)
            print("datasets shape:")
            print("{:^20}{:^20}{:^20}{:^20}".format("pan", "ms", "lms", "gt"))
            print(
                "{:^20}{:^20}{:^20}{:^20}".format(
                    str(self.pan.shape),
                    str(self.ms.shape),
                    str(self.lms.shape),
                    str(self.gt.shape),
                )
            )
        else:
            self.ms, self.lms, self.pan = self.get_divided(d, True)
            print("datasets shape:")
            print("{:^20}{:^20}{:^20}".format("pan", "ms", "lms"))
            print(
                "{:^20}{:^20}{:^20}".format(
                    str(self.pan.shape), str(self.ms.shape), str(self.lms.shape)
                )
            )

        self.size = self.ms.shape[0]

        # highpass filter
        self.hp = hp
        self.hp_ksize = hp_ksize
        if hp and hp_ksize is not None:
            self.group_high_pass(hp_ksize)

        # to tensor
        if norm_range:

            def norm_func(x):
                # return torch.tensor(x) / 2047.
                return torch.tensor(x, dtype=torch.float32) / const

        else:

            def norm_func(x):
                return torch.tensor(x, dtype=torch.float32)

        self.pan = norm_func(self.pan)
        self.ms = norm_func(self.ms)
        self.lms = norm_func(self.lms)
        if not full_res:
            self.gt = norm_func(self.gt)

        # geometrical transformation
        self.aug_prob = aug_prob
        self.geo_trans = (
            T.Compose(
                [T.RandomVerticalFlip(p=aug_prob), T.RandomHorizontalFlip(p=aug_prob)]
            )
            if aug_prob != 0.0
            else Identity()
        )

    @staticmethod
    def get_divided(d, full_resolution=False):
        if not full_resolution:
            return (
                np.asarray(d["gt"]),
                np.asarray(d["ms"]),
                np.asarray(d["lms"]),
                np.asarray(d["pan"]),
            )
        else:
            return (np.asarray(d["ms"]), np.asarray(d["lms"]), np.asarray(d["pan"]))

    @staticmethod
    def _get_high_pass(data, k_size):
        for i, img in enumerate(data):
            hp = cv2.boxFilter(img.transpose(1, 2, 0), -1, k_size)
            if hp.ndim == 2:
                hp = hp[..., np.newaxis]
            data[i] = img - hp.transpose(2, 0, 1)
        return data

    def group_high_pass(self, k_size):
        self.ms = self._get_high_pass(self.ms, k_size)
        self.pan = self._get_high_pass(self.pan, k_size)

    def aug_trans(self, *data):
        data_list = []
        seed = torch.random.seed()
        for d in data:
            torch.manual_seed(seed)
            d = self.geo_trans(d)
            data_list.append(d)
        return data_list

    def __getitem__(self, item):
        if hasattr(self, "gt"):
            tuple_data = (self.pan[item], self.ms[item], self.lms[item], self.gt[item])
        else:
            tuple_data = (self.pan[item], self.ms[item], self.lms[item])
        return self.aug_trans(*tuple_data) if self.aug_prob != 0.0 else tuple_data

    def __len__(self):
        return self.size

    def __repr__(self):
        return (
            f"num: {self.size} \n "
            f"augmentation: {self.geo_trans} \n"
            f"get high pass ms and pan: {self.hp} \n "
            f"filter kernel size: {self.hp_ksize}"
        )
