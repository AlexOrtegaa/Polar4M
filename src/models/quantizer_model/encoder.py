from torch import nn
from .residual import ResidualStack

import torch



class Encoder(
    nn.Module
):
    """Encoder module for a VQ-VAE model.

    This module encodes the input data into a latent representation using a series of
    convolutional layers and residual blocks. It reduces the spatial dimensions of the input
    while increasing the number of feature channels, preparing the data for vector quantization.

    Attributes:
        conv_stack:
            A sequential container of layers that make up the encoder.
    """
    def __init__(
            self,
            in_channels = int,
            out_channels = int,
            residual_channels = int,
            num_residual_layers = int,
    ):
        """Initialize the encoder with the specified parameters.

        Initializes the encoder with the given number of input channels, output channels,
        residual channels, and the number of residual layers.

        Args:
            in_channels:
                The number of input channels to the encoder.
            out_channels:
                The number of output channels from the encoder.
            residual_channels:
                The number of channels in the residual blocks.
            num_residual_layers:
                The number of residual layers in the residual stack.
        """
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
        return

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the encoder.

        This method takes the input data and encodes it into a latent representation
        by passing it through the convolutional layers and residual blocks.

        Args:
            x:
                A tensor representing the input data. The shape should be (batch_size, in_channels, height, width).

        Returns:
            A tensor containing the encoded latent representation. The shape will be (batch_size, out_channels, height/2**3, width/2**3).
        """
        return self.conv_stack(x) # spatial size reduced in half

