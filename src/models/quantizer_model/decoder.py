from torch import nn
from .residual import ResidualStack

import torch



class Decoder(
    nn.Module
):
    """Decoder module for a VQ-VAE model.

    This module reconstructs the input data from the latent representations
    produced by the encoder. It consists of a series of transposed convolutional layers
    and residual blocks to upsample the latent features back to the original input dimensions.

    Attributes:
        inv_conv_stack:
            A sequential container of layers that make up the decoder.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            residual_channels: int,
            num_residual_layers: int,
    ) -> None:
        """Initialize the decoder with the specified parameters.

        Initializes the decoder with the given number of input channels, output channels,
        residual channels, and the number of residual layers.

        Args:
            in_channels:
                The number of input channels to the decoder.
            out_channels:
                The number of output channels from the decoder.
            residual_channels:
                The number of channels in the residual blocks.
            num_residual_layers:
                The number of residual layers in the residual stack.
        """
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
        return

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the decoder.

        This method takes the latent representation as input and reconstructs the original data
        by passing it through the transposed convolutional layers and residual blocks.

        Args:
            x:
                A tensor containing the latent representation to be decoded.
                The shape of the tensor should be (batch_size, in_channels, height, width).

        Returns:
            A tensor containing the reconstructed data.
            The shape of the tensor will be (batch_size, out_channels, height*2**3, width*2**3).
        """
        return self.inv_conv_stack(x)


