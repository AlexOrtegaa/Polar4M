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
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size - 1, stride - 1, padding),
            nn.LayerNorm([in_channels, 32, 32]),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size - 1, stride - 1, padding),
            nn.LayerNorm([in_channels//2, 32, 32]),
            nn.GELU(),

            nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size, stride, padding),
            nn.LayerNorm([out_channels, 64, 64]),
        )

    def forward(self, x):
        return self.inv_conv_stack(x)


