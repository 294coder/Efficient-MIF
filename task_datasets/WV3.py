import torch
import torch.utils.data as data
import torchvision.transforms as T
import cv2
import numpy as np
import h5py
from typing import List, Tuple, Optional, Union

class Identity:
    def __call__(self, *args):
        return args

class WV3Datasets(data.Dataset):
    def __init__(
        self,
        file: Union[h5py.File, str, dict],
        aug_prob=0.0,
        hp=False,
        hp_ksize=(5, 5),
        norm_range=True,
        full_res=False,
    ):
        """

        :param d: h5py.File or dict
        :param aug_prob: augmentation probability
        :param hp: high pass for ms and pan. x = x - cv2.boxFilter(x)
        :param hp_ksize: cv2.boxFiler kernel size
        :param norm_range: normalize data range
        """
        super(WV3Datasets, self).__init__()
        # FIXME: should pass @path rather than @file which is h5py.File object to avoid can not be pickled error
        if isinstance(file, (str, h5py.File)):
            if isinstance(file, str):
                file = h5py.File(file)
            print(
                "warning: when @file is a h5py.File object, it can not be pickled.",
                "try to set DataLoader number_worker to 0",
            )
        if not full_res:
            self.gt, self.ms, self.lms, self.pan = self.get_divided(file)
            print("datasets shape:")
            print("{:^20}{:^20}{:^20}{:^20}".format("pan", "ms", "lms", "gt"))
            print(
                "{:^20}{:^20}{:^20}{:^20}".format(
                    str(tuple(self.pan.shape)),
                    str(tuple(self.ms.shape)),
                    str(tuple(self.lms.shape)),
                    str(tuple(self.gt.shape)),
                )
            )
        else:
            self.ms, self.lms, self.pan = self.get_divided(file, True)
            print("datasets shape:")
            print("{:^20}{:^20}{:^20}".format("pan", "ms", "lms"))
            print(
                "{:^20}{:^20}{:^20}".format(
                    str(tuple(self.pan.shape)),
                    str(tuple(self.ms.shape)),
                    str(tuple(self.lms.shape)),
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
                x = x / 2047.0
                return x

        else:

            def norm_func(x):
                return x

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
                torch.tensor(d["gt"][:], dtype=torch.float32),
                torch.tensor(d["ms"][:], dtype=torch.float32),
                torch.tensor(d["lms"][:], dtype=torch.float32),
                torch.tensor(d["pan"][:], dtype=torch.float32),
            )
        else:
            return (
                torch.tensor(d["ms"][:], dtype=torch.float32),
                torch.tensor(d["lms"][:], dtype=torch.float32),
                torch.tensor(d["pan"][:], dtype=torch.float32),
            )

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


def make_datasets(
    path, split_ratio=0.8, hp=True, seed=2022, aug_probs: Tuple = (0.0, 0.0)
):
    """
    if your dataset didn't split before, use this function will split your dataset into two part,
    which are train and validate datasets.
    :param device: device
    :param path: datasets path
    :param split_ratio: train validate split ratio
    :param hp: get high pass data, only works for ms and pan data
    :param seed: split data random state
    :param aug_probs: augmentation probabilities, type List
    :return: List[datasets]
    """
    d = h5py.File(path)
    ds = [
        torch.tensor(d["gt"]),
        torch.tensor(d["ms"]),
        torch.tensor(d["lms"]),
        torch.tensor(d["pan"]),
    ]
    n = ds[0].shape[0]
    s = int(n * split_ratio)
    random_perm = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(random_perm)

    train_set = {}
    val_set = {}
    for i, name in enumerate(["gt", "ms", "lms", "pan"]):
        ds[i] = ds[i][random_perm]
        train_set[name] = ds[i][:s]
        val_set[name] = ds[i][s:]
    train_ds = WV3Datasets(train_set, hp=hp, aug_prob=aug_probs[0])
    val_ds = WV3Datasets(val_set, hp=hp, aug_prob=aug_probs[1])
    return train_ds, val_ds

