from torch.functional import F

import torch.nn as nn


class ResidualLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            res_channels
    ):
        super().__init__()

        kernel_size = 3
        stride = 1
        padding = 1

        self.res_block = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=res_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(res_channels), # assuming input is 8x8 spatial size
            nn.GELU(),
            nn.Conv2d(in_channels=res_channels,
                      out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels), # assuming input is 8x8 spatial size
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x

class ResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, res_channels, num_residual_layers):
        super().__init__()
        # keep the same spatial size
        modules = [ResidualLayer(in_channels=in_channels, out_channels=out_channels, res_channels=res_channels) for _ in range(num_residual_layers)]
        self.residual_stack = nn.Sequential(*modules)

    def forward(self, x):
        x = self.residual_stack(x)
        return F.gelu(x)
