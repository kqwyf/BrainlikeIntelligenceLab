# Topic D2

The project is inspired by ISBI 2019 SegTHOR Challenge and designed for students to learn to build a model for a specific task. You are required to build a model for **multiple organ segmentation** from CT slices. You are encouraged to use *bi-directional network proposed in LMSER* to get better performance.

Data set: The ISBI 2019 SegTHOR Dataset can be downloaded from [here](https://jbox.sjtu.edu.cn/l/noXQob). The data description is same with ISBI Challenge. Every scan has a size of 512x512x(150x284) voxels. And the whole dataset includes 40 cases, **1-30 is train set and 31-40 is test set**, you are required train you model on train set and test your model on test set. You may need some pre-processing for those data.

Note: Maybe the dataset for this topic a little bit large for you and training may take
about 2 days.

## 思路

模型输出两个结果，一个是LMSER的重建结果，另一个是网络的预测（分类）概率输出。

考虑到CT平扫图存在时序性，那么我们不妨……
