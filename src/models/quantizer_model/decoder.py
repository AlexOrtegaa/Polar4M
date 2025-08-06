from torch import nn
from .residual import ResidualStack


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, residual_channels, num_residual_layers):
        super().__init__()

        kernel_size=4
        stride=2
        padding=1

        self.inv_conv_stack = nn.Sequential(
            ResidualStack(in_channels, in_channels, residual_channels, num_residual_layers),

            nn.GELU(),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(in_channels),    # 16x16 spatial size
            nn.GELU(),
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size, stride, padding),
            nn.BatchNorm2d(in_channels//2),   # 32x32 spatial size
            nn.GELU(),

            nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),      # 64x64 spatial size
        )

    def forward(self, x):
        return self.inv_conv_stack(x)


