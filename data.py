"""
读取nii格式: img = nib.load(path-to-img)

每个Patient_%2d目录下有两个文件:
* GT.nii            shape: (512, 512, n)    dtype: uint8
* Patient_%2d.nii   shape: (512, 512, n)    dtype: int32

GT.nii应该是Ground-Truth，因此数据更小。
"""
import os
from glob import glob

import nibabel as nib  # 读取nii格式
import numpy as np
import torch
from torch.utils.data import Dataset


class SegDataSet(Dataset):
    def __init__(self, path: str):
        img_files = sorted(glob(os.path.join(path, "*/Patient_*.nii")))
        gt_files = sorted(glob(os.path.join(path, "*/GT.nii")))
        self.imgs = []
        self.gts = []

        for img, gt in zip(img_files, gt_files):
            img_data = np.asanyarray(nib.load(img).dataobj)
            gt_data = np.asanyarray(nib.load(gt).dataobj)

            for i in range(img_data.shape[2]):
                self.imgs.append(img_data[:, :, i])
                self.gts.append(gt_data[:, :, i])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = torch.Tensor(self.imgs[index])
        gt = torch.Tensor(self.gts[index])
        return img, gt
