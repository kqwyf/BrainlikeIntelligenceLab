""" Fantastic Models """
from typing import List, Tuple

import torch
import torch.nn as nn


class ProjLayer(nn.Module):
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(dim1, dim2))

    def forward(self, x: torch.Tensor, go_up: bool):
        return x.matmul(self.proj) if go_up else x.matmul(self.proj.t())


class LMSER(nn.Module):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--lmser-steps", default=-1, type=int,
                help="LMSER的反复迭代次数。")
        parser.add_argument("--lmser-dims", required=True, nargs='+', type=int,
                help="LMSER的各层宽度，以逗号分隔")

    def __init__(self, args):
        super().__init__()
        self.steps = args.lmser_steps
        self.dims = args.lmser_dims

        self.layers = nn.ModuleList(ProjLayer(d1, d2) for d1, d2 in zip(self.dims[:-1], self.dims[1:]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 3, f"Dimension of input should be 3. Got {x.shape}"

        batch = x.shape[0]

        # Record inputs from above and below and only focus on hidden layers.
        emfs = [[torch.zeros(batch, dim).to(x.device), torch.zeros(batch, dim).to(x.device)]
                for dim in self.hidden_dims]
        emfs[0][0] = self.layers[0](x.view(batch, -1), True)

        last_out = None  # Record the last output
        step = 0

        while True:
            if step == self.steps:
                break
            step += 1

            _emfs = [self.sigmoid(emf[0] + emf[1]) for emf in emfs]  # n_hidden

            # Up stream
            ups = [layer(emf, True) for layer, emf in zip(self.layers[1:], _emfs[:-1])]
            for emf, up in zip(emfs[1:], ups):
                emf[0] = up

            # Down stream
            downs = [layer(emf, False) for layer, emf in zip(self.layers[1:], _emfs[1:])]
            for emf, down in zip(emfs[:-1], downs):
                emf[1] = down

            current_out = self.layers[0](_emfs[0], False)

            if last_out is not None and (last_out - current_out).detach().abs().mean() < 1e-5:
                break
            last_out = current_out

        return last_out


class SegModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor):
        pass


def _test():
    b, _shape = 4, (32, 48)
    _hidden_dims = [64, 32, 16]
    lmser = LMSER({
        "lmser_steps": -1,
        "lmser_dims": [64, 32, 16]
    })

    inp_img = torch.rand(b, *_shape)
    out_img = lmser(inp_img)

    print(out_img.shape)


if __name__ == "__main__":
    _test()
