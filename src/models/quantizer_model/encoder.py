from torch import nn
from .residual import ResidualStack


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=64,
            residual_channels=32,
            num_residual_layers=10,
    ):
        super().__init__()

        kernel_size=4
        stride=2
        padding=1

        self.conv_stack = nn.Sequential(
            # reduce in half the spatial size
            nn.Conv2d(in_channels, out_channels // 2, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels // 2),   # 32x32 spatial size
            nn.GELU(),
            # reduce in half the spatial size
            nn.Conv2d(out_channels // 2, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),       # 16x16 spatial size
            nn.GELU(),

            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),       # 8x8 spatial size

            ResidualStack(out_channels, out_channels, residual_channels, num_residual_layers),
        )

    def forward(self, x):
        return self.conv_stack(x) # spatial size reduced in half

