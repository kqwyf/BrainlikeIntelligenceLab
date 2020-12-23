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

from preprocessing import preprocess


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
        parser.add_argument("--dataset-path-train", required=True,
                help="训练集路径。")
        parser.add_argument("--dataset-path-test", required=True,
                help="测试集路径。")
        parser.add_argument("--dataset-num-dev-samples", type=int, required=True,
                help="将训练集中的多少（病人）样本作为验证集。")
        parser.add_argument("--dataset-preprocessing", action="append", choices=["add_channel_dim", "add_context1", "add_context2", "expand", "normalize", "normalize_group", "normalize_01"],
                help="""数据集需进行的预处理步骤，可用逗号隔开以连续使用多个步骤。
                        add_channel_dim：为每张图片加入channel维。
                        add_context1：为每张图片加入channel维，并对每组数据将前后1帧数据与当前帧在channel维上concat。
                        add_context2：为每张图片加入channel维，并对每组数据将前后2帧数据与当前帧在channel维上concat。
                        expand：将成组数据展开为独立的图片。由于输出图片不再成组，应放在所有预处理步骤的最后使用。
                        normalize：对每张图片进行正态归一化。
                        normalize_group：对每组图片进行正态归一化。
                        normalize_01：对每组图片归一化至[0, 1]。""")

    def __init__(self, args, mode):
        """
        :param mode: 可选项：train, dev, test
        """
        self.num_dev_samples = args.dataset_num_dev_samples

        if mode in ["train", "dev"]:
            path = args.dataset_path_train
        else:
            path = args.dataset_path_test
        img_files = sorted(glob(os.path.join(path, "*/Patient_*.nii")))
        gt_files = sorted(glob(os.path.join(path, "*/GT.nii")))

        if mode == "train":
            img_files, gt_files = img_files[:len(img_files) - self.num_dev_samples], gt_files[:len(img_files) - self.num_dev_samples]
        elif mode == "dev":
            img_files, gt_files = img_files[-self.num_dev_samples:], gt_files[-self.num_dev_samples:]

        self.imgs = []
        self.gts = []

        for img, gt in zip(img_files, gt_files):
            img_data = np.asanyarray(nib.load(img).dataobj)
            gt_data = np.asanyarray(nib.load(gt).dataobj)

            buff_img, buff_gt = [], []

            for i in range(img_data.shape[2]):
                buff_img.append(torch.tensor(img_data[:, :, i], dtype=torch.float32))
                buff_gt.append(torch.tensor(gt_data[:, :, i], dtype=torch.long))

            self.imgs.append(buff_img)
            self.gts.append(buff_gt)

        self.imgs, self.gts = preprocess(self.imgs, self.gts, args.dataset_preprocessing)
        self.group = (type(self.imgs[0]) != torch.Tensor) # 最终数据是否成组

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.group:
            img = torch.stack(self.imgs[index])
            gt = torch.stack(self.gts[index])
        else:
            img = self.imgs[index]
            gt = self.gts[index]
        return img, gt
