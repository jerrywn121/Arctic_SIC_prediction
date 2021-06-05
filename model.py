import torch.nn as nn
from utils import unfold_StackOverChannel, fold_tensor
from convlstm_net import ConvLSTM
import torch


class IceNet(nn.Module):
    def __init__(self, configs, land_mask):
        super().__init__()
        assert len(land_mask.shape) == 2
        input_dim = configs.input_dim * configs.patch_size[0] * configs.patch_size[1]
        output_dim = configs.output_dim * configs.patch_size[0] * configs.patch_size[1]
        self.base_net = ConvLSTM(input_dim, configs.hidden_dim, output_dim,
                                 None, configs.kernel_size)
        self.patch_size = configs.patch_size
        self.img_size = configs.img_size
        self.land_mask = unfold_StackOverChannel(torch.from_numpy(land_mask[None, None]), kernel_size=self.patch_size).to(configs.device)

    def forward(self, x):
        y = self.base_net(unfold_StackOverChannel(x, kernel_size=self.patch_size), self.land_mask)
        y = fold_tensor(y, output_size=self.img_size, kernel_size=self.patch_size)

        return y
