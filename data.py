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
from torch.utils.data import Dataset


class SegDataSet(Dataset):
    def __init__(self, path: str):
        self.img_files = sorted(glob(os.path.join(path, "*/Patient_*.nii")))
        self.gt_files = sorted(glob(os.path.join(path, "*/GT.nii")))

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
