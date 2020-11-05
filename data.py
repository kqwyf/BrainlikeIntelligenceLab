"""
读取nii格式: img = nib.load(path-to-img)

每个Patient_%2d目录下有两个文件:
* GT.nii            shape: (512, 512, n)    dtype: uint8
* Patient_%2d.nii   shape: (512, 512, n)    dtype: int32

GT.nii应该是Ground-Truth，因此数据更小。
GT的数据在0~4之间，是分类编号
"""
import os
from glob import glob

import nibabel as nib  # 读取nii格式
import numpy as np
import torch
from torch.utils.data import Dataset


class SegDataSet(Dataset):
    """
    读取CT数据。
    数据目录组织形式：
    .
    ├── Patient_01
    │   ├── GT.nii
    │   └── Patient_01.nii
    ......
    └── Patient_n
        ├── GT.nii
        └── Patient_n.nii
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--dataset-path", required=True,
                help="数据集路径。")
        parser.add_argument("--dataset-group", type=bool, default=True,
                help="是否将同一patient的图片成组输入。")

    def __init__(self, args):
        img_files = sorted(glob(os.args.path.join(args.path, "*/Patient_*.nii")))
        gt_files = sorted(glob(os.args.path.join(args.path, "*/GT.nii")))
        self.imgs = []
        self.gts = []

        for img, gt in zip(img_files, gt_files):
            img_data = np.asanyarray(nib.load(img).dataobj)
            gt_data = np.asanyarray(nib.load(gt).dataobj)

            buff_img, buff_gt = [], []

            for i in range(img_data.shape[2]):
                buff_img.append(img_data[:, :, i])
                buff_gt.append(gt_data[:, :, i])

            if args.group:
                self.imgs.append(buff_img)
                self.gts.append(buff_gt)
            else:
                self.imgs.extend(buff_img)
                self.gts.extend(buff_gt)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = torch.Tensor(self.imgs[index])
        gt = torch.Tensor(self.gts[index])
        return img, gt
