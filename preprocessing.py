import torch


def add_context1(imgs: list, gts: list):
    imgs_new = []
    for img_list in imgs:
        img_list_new = [torch.cat((
            torch.zeros(img_list[0].shape).unsqueeze(0),
            img_list[0].unsqueeze(0),
            img_list[1].unsqueeze(0)
        ))]
        for i in range(1, len(img_list) - 1):
            img_list_new.append(torch.cat((
                img_list[i - 1].unsqueeze(0),
                img_list[i].unsqueeze(0),
                img_list[i + 1].unsqueeze(0)
            )))
        img_list_new.append(torch.cat((
            img_list[-2].unsqueeze(0),
            img_list[-1].unsqueeze(0),
            torch.zeros(img_list[0].shape).unsqueeze(0)
        )))
        imgs_new.append(img_list_new)
    return imgs_new, gts


def add_context2(imgs: list, gts: list):
    imgs_new = []
    for img_list in imgs:
        img_list_new = [torch.cat((
            torch.zeros(img_list[0].shape).unsqueeze(0),
            torch.zeros(img_list[0].shape).unsqueeze(0),
            img_list[0].unsqueeze(0),
            img_list[1].unsqueeze(0),
            img_list[2].unsqueeze(0)
        ))]
        img_list_new.append(torch.cat((
            torch.zeros(img_list[0].shape).unsqueeze(0),
            img_list[0].unsqueeze(0),
            img_list[1].unsqueeze(0),
            img_list[2].unsqueeze(0),
            img_list[3].unsqueeze(0)
        )))
        for i in range(2, len(img_list) - 2):
            img_list_new.append(torch.cat((
                img_list[i - 2].unsqueeze(0),
                img_list[i - 1].unsqueeze(0),
                img_list[i].unsqueeze(0),
                img_list[i + 1].unsqueeze(0),
                img_list[i + 2].unsqueeze(0)
            )))
        img_list_new.append(torch.cat((
            img_list[-4].unsqueeze(0),
            img_list[-3].unsqueeze(0),
            img_list[-2].unsqueeze(0),
            img_list[-1].unsqueeze(0),
            torch.zeros(img_list[0].shape).unsqueeze(0)
        )))
        img_list_new.append(torch.cat((
            img_list[-3].unsqueeze(0),
            img_list[-2].unsqueeze(0),
            img_list[-1].unsqueeze(0),
            torch.zeros(img_list[0].shape).unsqueeze(0),
            torch.zeros(img_list[0].shape).unsqueeze(0)
        )))
        imgs_new.append(img_list_new)
    return imgs_new, gts


def expand(imgs: list, gts: list):
    imgs = [im for img_list in imgs for im in img_list]
    gts = [gt for gt_list in gts for gt in gt_list]
    return imgs, gts


def normalize(imgs: list, gts: list):
    for img_list in imgs:
        for i in range(len(img_list)):
            img_list[i] = (img_list[i] - img_list[i].mean()) / \
                img_list[i].var().sqrt()
    return imgs, gts


def normalize_group(imgs: list, gts: list):
    for i in range(len(imgs)):
        img_tensor = torch.stack(imgs[i])
        img_tensor = (img_tensor - img_tensor.mean()) / img_tensor.var().sqrt()
        imgs[i] = [im for im in img_tensor]
    return imgs, gts


PREPROCESSING_FUNCS = {
    "add_context1": add_context1,
    "add_context2": add_context2,
    "expand": expand,
    "normalize": normalize,
    "normalize_group": normalize_group,
}

# 在DataLoader的collate_fn中调用以下函数以进行自定义预处理操作


def preprocess(imgs, gts, tasks):
    # TODO: 考虑更好的归一化方式？
    for task in tasks:
        imgs, gts = PREPROCESSING_FUNCS[task](imgs, gts)
    return imgs, gts
