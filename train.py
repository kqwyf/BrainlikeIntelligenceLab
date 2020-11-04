import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import directed_hausdorff, dice

from data import SegDataSet
from lmser import LMSER

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
    def __init__(self):
        self.dice_scores = []
        self.ahd_scores = []

    def update(self, pred_batch: torch.Tensor, target_batch: torch.Tensor):
        assert len(pred_batch) == len(target_batch)
        for i in range(len(pred_batch)):
            dice_score = []
            for j in range(5): # 5是分类数
                dice_score.append(dice((pred_batch[i] == j).flatten(), (target_batch[i] == j).flatten()))
                # TODO: 计算AHD。scipy中的directed_hausdorff计算的是HD而不是AHD。相关资料参考（关键词：Hausdorff）https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/
            self.dice_scores.append(tuple(dice_score))

    def result(self):
        dice_mean = np.mean(np.array(self.dice_scores), axis=0)
        dice_var = np.var(np.array(self.dice_scores), axis=0)
        return dice_mean, dice_var

class Trainer:
    def __init__(self, model, optimizer, rebuild_criterion, classify_criterion, alpha: float, lmser_steps: int):
        """
        :param lmser_steps: LMSER的反复迭代次数
        :param alpha: 重建loss的系数
        """
        self.model = model
        self.optimizer = optimizer
        self.rebuild_criterion = rebuild_criterion
        self.classify_criterion = classify_criterion
        self.lmser_steps = lmser_steps
        self.alpha = alpha

    def train(self, train_data_loader, dev_data_loader, num_epochs):
        # NOTE: 我们希望model.forward()返回两个对象，一个是LMSER的重建结果，另一个是网络的预测（分类）概率输出
        # 期望重建输出形状与输入相同，即: (512, 512, n)
        # 期望预测输出形状: (512, 512, n, 5)
        for epoch_i in range(1, num_epochs + 1):
            # train
            self.model.train()
            self.rebuild_criterion.train()
            self.classify_criterion.train()
            for iter_i, (data_batch, target_batch) in enumerate(train_data_loader):
                # forward
                rebuild_out, classify_out = self.model(data_batch)
                rebuild_loss = self.rebuild_criterion(rebuild_out, data_batch) # TODO
                classify_loss = self.classify_criterion(classify_out, target_batch) # TODO: 检查参数列表
                loss = self.alpha * rebuild_loss + classify_loss
                # TODO: 输出loss
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # validation for every epoch
            self.model.eval()
            self.rebuild_criterion.eval()
            self.classify_criterion.eval()
            metric = Metric()
            with torch.no_grad():
                for iter_i, (data_batch, target_batch) in enumerate(dev_data_loader):
                    rebuild_out, classify_out = self.model(data_batch)
                    metric.update(classify_out.cpu().detach().numpy().argmax(axis=4), target_batch.cpu().detach().numpy())
                dice_mean, dice_var = metric.result()
                # TODO: 输出validation结果
                # TODO: 是否加入scheduler?


    def test(self):
        pass


if __name__ == "__main__":
    pass
