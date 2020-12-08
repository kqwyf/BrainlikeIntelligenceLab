import logging
import os
import sys
from glob import glob

import configargparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import SegDataSet
from model import SegModel

CONFIG_FILE = "conf/train.yaml"  # 默认配置文件路径
LOG_FILENAME = "cut.log"
CKPT_FILENAME_GLOB = "checkpoint.ep.*"


def load_checkpoint(path: str, model: nn.Module):
    logging.info(f"Load checkpoint from {path}")
    _, model_state, _, _ = torch.load(path)
    model.load_state_dict(model_state)


@torch.no_grad()
def cutter(args, model: nn.Module, data_loader: DataLoader):
    model.eval()

    i = 0
    with tqdm(desc="Seg...") as pbar:
        for iter_i, (data_batch, target_batch) in enumerate(data_loader):
            data_batch = data_batch.to(args.device)
            out = model(data_batch)                 # shape: [B, C, H, W]
            out = out.permute(0, 2, 3, 1)           # shape: [B, H, W, C]
            out = out.max(dim=-1)[1].cpu().numpy()  # shape: [B, H, W, 1]

            # 保存结果
            for img in out:
                im_path = os.path.join(args.seg_dir, f"{i}.jpg")
                plt.imsave(im_path, img)
                i += 1
                pbar.update()

    logging.info("Segmentation completed.")


def main(cmd_args):
    parser = configargparse.ArgumentParser(
        default_config_files=[CONFIG_FILE],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter
    )
    parser.add("--config", is_config_file=True, default=CONFIG_FILE, help="配置文件。")
    parser.add_argument("--device", choices=["cuda:0", "cpu"], default="cuda:0", help="使用CPU或GPU进行训练。")
    parser.add_argument("--exp-dir", default=None, help="日志、模型等文件的存放路径，默认为exp/{exp_name}。其中{exp_name}为配置文件名去除后缀。")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size。")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint存储位置。")

    SegModel.add_arguments(parser)
    SegDataSet.add_arguments(parser)

    args, _ = parser.parse_known_args(cmd_args)    # 只加载认识的参数
    parser.print_values()

    # 目录创建
    if args.exp_dir is None:
        args.exp_dir = "exp/{}".format(os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(args.exp_dir, exist_ok=True)

    args.seg_dir = os.path.join(args.exp_dir, "seg")
    os.makedirs(args.seg_dir, exist_ok=True)

    # 如果不提供checkpoint，则默认加载最后一轮训练结果
    if args.checkpoint is None:
        ckps = glob(os.path.join(args.exp_dir, CKPT_FILENAME_GLOB))
        if len(ckps) == 0:
            logging.error("No chekpoint found.")
            sys.exit(-1)
        ckpt_path = sorted(ckps)[-1]
        args.checkpoint = ckpt_path

    logging.basicConfig(filename=os.path.join(args.exp_dir, LOG_FILENAME),
                        level="INFO",
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%y-%m-%d %H:%M:%S")

    # Create testset
    dataset = SegDataSet(args, "test")
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    CT = SegModel(args).to(args.device)
    load_checkpoint(args.checkpoint, CT)

    # Segment the region
    cutter(args, CT, data_loader)


if __name__ == "__main__":
    main(sys.argv[1:])
