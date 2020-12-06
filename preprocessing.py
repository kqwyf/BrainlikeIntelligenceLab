import torch

def expand(batch):
    return batch

def group_diff(batch):
    return batch

def group_diff2(batch):
    return batch

PREPROCESSING_FUNCS = {
    "expand": expand,
    "group_diff": group_diff,
    "group_diff2": group_diff2,
}

# 在DataLoader的collate_fn中调用以下函数以进行自定义预处理操作
def preprocess(batch, tasks):
    batch = torch.tensor(batch)
    # TODO: 归一化至均值0，方差1，或者归一化至[-1,1]区间
    for task in tasks:
        batch = PREPROCESSING_FUNCS[task](batch)
    return batch

