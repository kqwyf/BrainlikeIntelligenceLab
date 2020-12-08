""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--unet-input-width", default=512, type=int,
                help="输入宽度。")
        parser.add_argument("--unet-input-height", default=512, type=int,
                help="输入高度。")
        parser.add_argument("--unet-input-channels", default=3, type=int,
                help="输入channel数。")
        parser.add_argument("--unet-init-channels", default=64, type=int,
                help="UNet起始channel数（也是最终channel数）。")
        parser.add_argument("--unet-num-layers", default=4, type=int,
                help="UNet层数（单边）。")
        parser.add_argument("--unet-bilinear", default=True, type=bool,
                help="UNet中是否使用双线性插值进行上采样。")

    def __init__(self, args, num_classes: int):
        super(UNet, self).__init__()
        self.input_width = args.unet_input_width
        self.input_height = args.unet_input_height
        self.num_channels = args.unet_input_channels
        self.init_channels = args.unet_init_channels
        self.num_layers = args.unet_num_layers
        self.bilinear = args.unet_bilinear
        self.num_classes = num_classes

        self.inc = DoubleConv(self.num_channels, self.init_channels)
        current_num_channels = self.init_channels
        factor = 2 if self.bilinear else 1
        self.downs = []
        self.ups = []
        for i in range(self.num_layers - 1):
            self.downs.append(Down(current_num_channels, current_num_channels * 2))
            current_num_channels *= 2
        self.downs.append(Down(current_num_channels, current_num_channels * 2 // factor))
        self.downs = nn.ModuleList(self.downs)
        current_num_channels *= 2
        for i in range(self.num_layers - 1):
            self.ups.append(Up(current_num_channels, current_num_channels // (2 * factor), self.bilinear))
            current_num_channels //= 2
        self.ups.append(Up(current_num_channels, current_num_channels // 2, self.bilinear))
        self.ups = nn.ModuleList(self.ups)
        self.outc = OutConv(self.init_channels, num_classes)

    def forward(self, x):
        xs = [x]
        xs.append(self.inc(x))
        for i in range(self.num_layers):
            xs.append(self.downs[i](xs[-1]))
        for i in range(self.num_layers):
            xs[-(i + 2)] = self.ups[i](xs[-(i + 1)], xs[-(i + 2)])
        logits = self.outc(xs[1])
        return logits

