import logging
import os
import sys
from glob import glob
from typing import Optional

import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.core.umath_tests import inner1d
from scipy.spatial.distance import dice, directed_hausdorff

from data import SegDataSet
from model import SegModel

CONFIG_FILE = "conf/train.yaml" # 默认配置文件路径
LOG_FILENAME = "train.log"
CKPT_FILENAME = "checkpoint.ep.%02d" # checkpoint默认文件名，其中整数为已训练轮数
CKPT_FILENAME_GLOB = "checkpoint.ep.*"


class Metric:
    """
    包装了Dice和AHD两种衡量准确率的指标，它们是原数据集论文中使用的性能指标。在
    测试阶段前创建一个Metric对象，然后每次对一个batch数据进行预测后将预测结果和
    真实标签送入对象的update()方法，最后调用result()即可获得测试结果。

    Dice距离和AHD距离均调用scipy计算。

    Dice: 重叠区域面积的2倍除以预测区域与实际区域面积之和。也可等价地记为
    (2TP)/(2TP+FP+FN)，其中TP等分别表示true positive等。

    AHD: 计算点集A和B间的距离时，定义d(A,B)表示对A中每个点a，找到B中距离其最近的
    点b，计算(1/N)*\sum_{a}||a-b||。则AHD定义为max(d(A,B), d(B,A))
    """

    def __init__(self, args):
        self.num_classes = args.num_classes
        self.reset()

    def reset(self):
        self.dice_scores = []
        self.ahd_scores = []

    def update(self, pred_batch: np.ndarray, target_batch: np.ndarray):
        """
        Input:
          - pred_batch.shape:   [B, H, W]
          - target_batch.shape: [B, H, W]
        """
        assert len(pred_batch) == len(target_batch)

        for i in range(len(pred_batch)):
            dice_score = []
            ahd_score = []

            for j in range(self.num_classes):
                pred_bool = (pred_batch[i] == j).flatten()
                target_bool = (target_batch[i] == j).flatten()
                if np.alltrue(pred_bool == target_bool): # 规避dice()在输入全0时输出nan的问题
                    dice_score.append(1.0)
                else:
                    dice_score.append(1 - dice(pred_bool, target_bool)) # Dice score = 1 - Dice distance
                # 计算AHD
                if j == 0:
                    ahd_score.append(-1.0) # 不计算背景像素的分数，因为会占很大内存，而且没有用处
                else:
                    pred_coord = np.array(np.where(pred_batch[i] == j)).T
                    target_coord = np.array(np.where(target_batch[i] == j)).T
                    if len(target_coord) == 0 and len(pred_coord) == 0: # 规避0长度数组
                        ahd_score.append(1.0)
                    elif len(target_coord) == 0 or len(pred_coord) == 0:
                        ahd_score.append(0.0)
                    else:
                        D_mat = np.sqrt(inner1d(pred_coord, pred_coord)[np.newaxis].T
                                        + inner1d(target_coord, target_coord)
                                        - 2 * np.dot(pred_coord, target_coord.T))
                        dH = np.max(np.array([
                                np.max(np.min(D_mat, axis=0)),
                                np.max(np.min(D_mat, axis=1))
                            ]))
                        ahd_score.append(dH)

            self.dice_scores.append(tuple(dice_score))
            self.ahd_scores.append(tuple(ahd_score))

    def result(self):
        dice_mean = np.mean(np.array(self.dice_scores), axis=0)
        dice_var = np.var(np.array(self.dice_scores), axis=0)

        ahd_mean = np.mean(np.array(self.ahd_scores), axis=0)
        ahd_var = np.var(np.array(self.ahd_scores), axis=0)
        return dice_mean, dice_var, ahd_mean, ahd_var


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--focal-alpha", default=None, action="append", type=float,
                help="Focal Loss的alpha参数，控制各类权重。")
        parser.add_argument("--focal-gamma", default=0.0, type=float,
                help="Focal Loss的gamma参数。")

    def __init__(self, args, reduction: str = 'mean', ignore_index: int = -100):
        """
        Args:
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = torch.tensor(args.focal_alpha)
        self.gamma = args.focal_gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=self.alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: torch.FloatTensor, y: torch.LongTensor) -> torch.FloatTensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # The full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def one_hot(x, num_classes):
    result = torch.zeros(x.shape + (num_classes,)).to(x.device)
    result.scatter_(len(x.shape), x.unsqueeze(len(x.shape)), torch.ones(x.shape + (1,), device=x.device))
    return result


def save_checkpoint(path: str, epoch_i: int, model: nn.Module, criterion: nn.Module, optimizer: nn.Module):
    torch.save((epoch_i, model.state_dict(), criterion.state_dict(), optimizer.state_dict()), path)


def load_checkpoint(path: str, model: nn.Module, criterion: nn.Module, optimizer: nn.Module):
    epoch_i, model_state, criterion_state, optimizer_state = torch.load(path)
    model.load_state_dict(model_state)
    criterion.load_state_dict(criterion_state)
    optimizer.load_state_dict(optimizer_state)
    return epoch_i


class Trainer:
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--resume", default="NONE", nargs="?",
                help="从参数给出的checkpoint继续训练。“NONE”表示忽略该参数。若参数为空，则从exp_dir参数给出的路径中读取编号最大的checkpoint。")
        parser.add_argument("--num-epochs", required=True, type=int,
                help="训练轮数。")
        parser.add_argument("--accum-grad", type=int, default=1,
                help="连续累加几个batch的梯度，一并优化。若当前epoch剩余batch数不足，则在epoch结束时优化。")

    def __init__(self, args, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_classes = args.num_classes
        self.num_epochs = args.num_epochs
        self.accum_grad = args.accum_grad
        self.exp_dir = args.exp_dir
        self.device = args.device
        if args.dataset_num_dev_samples == 0:
            self.do_validation = False
        else:
            self.do_validation = True
        if args.resume == "NONE": # 没有使用resume参数
            resume = False
            self.epoch_start = 0
        else: # 使用了resume参数
            if args.resume is None: # 使用了resume参数，没有填写checkpoint路径
                resume = True
                ckpt_path = sorted(glob(os.path.join(args.exp_dir, CKPT_FILENAME_GLOB)))[-1]
            else: # 使用了resume参数，填写了checkpoint路径
                resume = True
                ckpt_path = args.resume
            self.epoch_start = load_checkpoint(ckpt_path, self.model, self.criterion, self.optimizer)
        self.metric = Metric(args)

    def train(self, data_loader_train, data_loader_dev):
        for epoch_i in range(self.epoch_start + 1, self.num_epochs + 1):
            # train
            logging.info("Epoch %d/%d: Training" % (epoch_i, self.num_epochs))
            self.model.train()
            self.criterion.train()
            self.optimizer.zero_grad()

            num_iters = len(data_loader_train)
            for iter_i, (data_batch, target_batch) in enumerate(data_loader_train):
                data_batch, target_batch = data_batch.to(self.device), target_batch.to(self.device)
                # forward
                out = self.model(data_batch)
                out_p = out.permute(0, 2, 3, 1) # 将channel维放在最后
                if type(self.criterion) in [nn.MSELoss]:
                    loss = self.criterion(out_p.reshape(-1, out_p.shape[-1]), one_hot(target_batch.flatten(), self.num_classes))
                else:
                    loss = self.criterion(out_p.reshape(-1, out_p.shape[-1]), target_batch.flatten())
                logging.info("    Iter %d/%d: loss = %.3f" % (iter_i + 1, num_iters, loss))
                # backward
                loss.backward()
                # 梯度累加
                if iter_i % self.accum_grad == self.accum_grad - 1 or iter_i == len(data_loader_train) - 1:
                    if iter_i % self.accum_grad != self.accum_grad - 1:
                        logging.warn("    Force accumulating gradients at epoch end.")
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # validation
            if self.do_validation:
                logging.info("Epoch %d/%d: Validation" % (epoch_i, self.num_epochs))
                self.model.eval()
                self.criterion.eval()
                self.metric.reset()
                with torch.no_grad():
                    for iter_i, (data_batch, target_batch) in enumerate(data_loader_dev):
                        data_batch, target_batch = data_batch.to(self.device), target_batch.to(self.device)
                        out = self.model(data_batch)        # shape: [B, C, H, W]
                        out_p = out.permute(0, 2, 3, 1)     # shape: [B, H, W, C]
                        self.metric.update(out_p.cpu().detach().numpy().argmax(axis=3),  # shape: [B, H, W]
                                           target_batch.cpu().detach().numpy())
                dice_mean, dice_var, ahd_mean, ahd_var = self.metric.result()
                dice_stdvar, ahd_stdvar = np.sqrt(dice_var), np.sqrt(ahd_var)
                logging.info("    Dice = (%.2f +- %.2f, %.2f +- %.2f, %.2f +- %.2f, %.2f +- %.2f, %.2f +- %.2f)" %
                        (dice_mean[0], dice_stdvar[0],
                         dice_mean[1], dice_stdvar[1],
                         dice_mean[2], dice_stdvar[2],
                         dice_mean[3], dice_stdvar[3],
                         dice_mean[4], dice_stdvar[4],))
                logging.info("    AHD = (%.2f +- %.2f, %.2f +- %.2f, %.2f +- %.2f, %.2f +- %.2f, %.2f +- %.2f)" %
                        (ahd_mean[0], ahd_stdvar[0],
                         ahd_mean[1], ahd_stdvar[1],
                         ahd_mean[2], ahd_stdvar[2],
                         ahd_mean[3], ahd_stdvar[3],
                         ahd_mean[4], ahd_stdvar[4],))
                # TODO: 是否加入scheduler?

            # 保存checkpoint
            ckpt_path = os.path.join(self.exp_dir, CKPT_FILENAME % epoch_i)
            save_checkpoint(ckpt_path, epoch_i, self.model, self.criterion, self.optimizer)
            logging.info("Checkpoint saved to " + ckpt_path)

        logging.info("Training finished.")


def main(cmd_args):
    parser = configargparse.ArgumentParser(
                default_config_files=[CONFIG_FILE],
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter
            )
    parser.add("--config", is_config_file=True, default=CONFIG_FILE,
            help="配置文件。")
    parser.add_argument("--device", choices=["cuda:0", "cpu"], default="cuda:0",
            help="使用CPU或GPU进行训练。")
    parser.add_argument("--exp-dir", default=None,
            help="日志、模型等文件的存放路径，默认为exp/{exp_name}。其中{exp_name}为配置文件名去除后缀。")
    parser.add_argument("--batch-size", type=int, default=1,
            help="Batch size。")
    parser.add_argument("--optimizer", choices=["adam"], default="adam",
            help="优化器，可选项包括：adam。")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
            help="学习率。")
    parser.add_argument("--criterion", choices=["ce", "mse", "focal"], default="focal",
            help="Loss函数，可选项包括：ce（交叉熵），mse（最小均方误差），focal（Focal Loss）。")

    FocalLoss.add_arguments(parser)
    Trainer.add_arguments(parser)
    SegModel.add_arguments(parser)
    SegDataSet.add_arguments(parser)

    args = parser.parse_args(cmd_args)

    if args.exp_dir is None:
        args.exp_dir = "exp/{}".format(os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(args.exp_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(args.exp_dir, LOG_FILENAME),
                        level="INFO",
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%y-%m-%d %H:%M:%S")

    # create objects needed by training
    dataset_train = SegDataSet(args, "train")
    dataset_dev= SegDataSet(args, "dev")
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    data_loader_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=args.batch_size, shuffle=False)

    model = SegModel(args).to(args.device)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise NotImplementedError("Optimizer %s not implemented" % args.optimizer)

    if args.criterion == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == "mse":
        criterion = nn.MSELoss()
    elif args.criterion == "focal":
        criterion = FocalLoss(args)
    else:
        raise NotImplementedError("Criterion %s not implemented" % args.criterion)
    criterion.to(args.device)

    trainer = Trainer(args, model, optimizer, criterion)
    trainer.train(data_loader_train, data_loader_dev)


if __name__ == "__main__":
    main(sys.argv[1:])
