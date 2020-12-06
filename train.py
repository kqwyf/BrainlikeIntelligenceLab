import sys
import os
from glob import glob
import configargparse
import logging
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import dice, directed_hausdorff

from data import SegDataSet
from model import SegModel
from preprocessing import preprocess


CONFIG_FILE = "conf/train.yaml" # 默认配置文件路径
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

    def update(self, pred_batch: torch.Tensor, target_batch: torch.Tensor):
        assert len(pred_batch) == len(target_batch)
        for i in range(len(pred_batch)):
            dice_score = []
            for j in range(self.num_classes):
                dice_score.append(dice((pred_batch[i] == j).flatten(), (target_batch[i] == j).flatten()))
                # TODO: 计算AHD。scipy中的directed_hausdorff计算的是HD而不是AHD。相关资料参考（关键词：Hausdorff）https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/
            self.dice_scores.append(tuple(dice_score))

    def result(self):
        dice_mean = np.mean(np.array(self.dice_scores), axis=0)
        dice_var = np.var(np.array(self.dice_scores), axis=0)
        return dice_mean, dice_var


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
        parser.add_argument("--num-epoch", required=True, type=int,
                help="训练轮数。")
        parser.add_argument("--accum-grad", type=int, default=1,
                help="连续累加几个batch的梯度，一并优化。若当前epoch剩余batch数不足，则在epoch结束时优化。")

    def __init__(self, args, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = args.num_epochs
        self.accum_grad = args.accum_grad
        self.exp_dir = args.exp_dir
        if args.resume == "": # 没有使用resume参数
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
            logging.info("Epoch %d: Training"%epoch_i)
            self.model.train()
            self.criterion.train()
            self.optimizer.zero_grad()
            for iter_i, (data_batch, target_batch) in enumerate(data_loader_train):
                # forward
                out = self.model(data_batch)
                loss = self.criterion(out.reshape(-1, out.shape[-1]), target_batch.flatten())
                logging.info("    loss = %.3f"%(loss))
                # backward
                loss.backward()
                # 梯度累加
                if iter_i % self.accum_grad == self.accum_grad - 1 or iter_i == len(data_loader_train) - 1:
                    if iter_i % self.accum_grad != self.accum_grad - 1:
                        logging.warn("    Force accumulating gradients at epoch end.")
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            # validation
            logging.info("Epoch %d: Validation"%epoch_i)
            self.model.eval()
            self.criterion.eval()
            self.metric.reset()
            with torch.no_grad():
                for iter_i, (data_batch, target_batch) in enumerate(data_loader_dev):
                    rebuild_out, classify_out = self.model(data_batch)
                    self.metric.update(classify_out.cpu().detach().numpy().argmax(
                        axis=4), target_batch.cpu().detach().numpy())
            dice_mean, dice_var = self.metric.result()
            logging.info("    Dice = (%.2f +- %.2f), AHD = (NULL)")
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
    parser.add("--config", is_config_file=True,
            help="配置文件。")
    parser.add_argument("--exp-dir", default=None,
            help="日志、模型等文件的存放路径，默认为exp/{exp_name}。其中{exp_name}为配置文件名去除后缀。")
    parser.add_argument("--batch-size", type=int, default=1,
            help="Batch size。")
    parser.add_argument("--optimizer", choices=["adam"], default="adam",
            help="优化器，可选项包括：adam。")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
            help="学习率。")
    parser.add_argument("--criterion", choices=["ce", "mse"], default="ce",
            help="Loss函数，可选项包括：ce（交叉熵），mse（最小均方误差）。")
    parser.add_argument("--preprocessing", action="append", choices=["expand", "group_diff", "group_diff2"],
            help="""数据输入模型前需进行的预处理步骤，可用逗号隔开以连续使用多个步骤。
                    归一化步骤将在下列所有步骤之前自动完成。
                    expand：将成组数据展开至batch_size维。
                    group_diff：对每组数据求1阶差分，与原数据在channel维上concat。
                    group_diff2：对每组数据求1阶和2阶差分，与原数据在channel维上concat。""")

    Trainer.add_arguments(parser)
    SegModel.add_arguments(parser)
    SegDataSet.add_arguments(parser)

    args, _ = parser.parse_args(cmd_args)

    if args.exp_dir is None:
        args.exp_dir = "exp/{}".format(os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(args.exp_dir, exist_ok=True)

    logging.basicConfig(filename=args.log_path,
                        format="%(asctime)s - $(levelname)s - %(message)s",
                        datefmt="%y-%m-%d %H:%M:%S")

    # create objects needed by training
    dataset_train = SegDataSet(args, "train")
    dataset_dev= SegDataSet(args, "dev")
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
            shuffle=True, collate_fn=lambda x: preprocess(x, args.preprocessing))
    data_loader_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=args.batch_size,
            shuffle=True, collate_fn=lambda x: preprocess(x, args.preprocessing))

    model = SegModel(args)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise NotImplementedError("Optimizer %s not implemented" % args.optimizer)

    if args.criterion == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == "mse":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Criterion %s not implemented" % args.criterion)

    trainer = Trainer(args, model, optimizer, criterion)
    trainer.train(data_loader_train, data_loader_dev)

if __name__ == "__main__":
    main(sys.argv[1:])

