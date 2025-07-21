from torch import nn
from .residual import ResidualStack


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, residual_channels, num_residual_layers):
        super().__init__()

        kernel_size=4
        stride=2
        padding=1

        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size, stride, padding),
            nn.GELU(),

            nn.Conv2d(out_channels // 2, out_channels, kernel_size - 1, stride - 1, padding),
            nn.GELU(),

            nn.Conv2d(out_channels, out_channels, kernel_size - 1, stride - 1, padding),

            ResidualStack(out_channels, out_channels, residual_channels, num_residual_layers),
        )

    def forward(self, x):
        return self.conv_stack(x)

