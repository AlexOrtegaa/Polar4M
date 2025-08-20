from torch.functional import F

import torch.nn as nn

import torch



class ResidualLayer(
    nn.Module
):
    """Residual layer for a neural network.

    This layer implements a residual block that applies two convolutional layers
    with GELU activation and batch normalization. The input is added to the output
    of the residual block to create a skip connection, allowing for better gradient flow
    and improved training of deep networks.

    Attributes:
        res_block:
            A sequential container of layers that make up the residual block.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            res_channels: int,
    ) -> None:
        """Initialize the residual layer with the specified parameters.

        Initializes the residual layer with the given number of input channels,
        output channels, and residual channels. The layer consists of two convolutional
        layers with GELU activation and batch normalization, allowing for a residual connection
        that adds the input to the output of the residual block.

        Args:
            in_channels:
                The number of input channels to the residual layer.
            out_channels:
                The number of output channels from the residual layer.
            res_channels:
                The number of channels in the residual block. The residual block is between
                the input and output channels, allowing for a flexible architecture that can
                adapt to different input and output dimensions.
        """
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
        return

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the residual layer.

        This method takes the input tensor, applies the residual block, and adds the input
        to the output of the residual block. This creates a skip connection.

        Args:
            x:
                A tensor containing the input data to the residual layer.
        Returns:
            A tensor containing the output of the residual layer after applying the residual block
            and adding the input to it.
        """

        x = x + self.res_block(x)
        return x

class ResidualStack(
    nn.Module
):
    """Stack of residual layers for a neural network.

    This stack consists of multiple residual layers, each defined by the `ResidualLayer` class.

    Attributes:
        residual_stack:
            A sequential container of residual layers that make up the stack.
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            res_channels,
            num_residual_layers
    ) -> None:
        """Initialize the residual stack with the specified parameters.

        Initializes the residual stack with the given number of input channels, output channels,
        residual channels, and the number of residual layers. Each layer in the stack is an instance
        of the `ResidualLayer` class. The stack maintains the same spatial size throughout.

        Args:
            in_channels:
                The number of input channels to the first residual layer.
            out_channels:
                The number of output channels from the last residual layer.
            res_channels:
                The number of channels in each residual block.
            num_residual_layers:
                The number of residual layers in the stack.
        """
        super().__init__()
        # keep the same spatial size
        modules = [ResidualLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            res_channels=res_channels
        ) for _ in range(num_residual_layers)]
        self.residual_stack = nn.Sequential(*modules)
        return

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the residual stack.

        This method takes the input tensor and passes it through the stack of residual layers.

        Args:
            x:
                A tensor containing the input data to the residual stack.
        """
        x = self.residual_stack(x)
        return F.gelu(x)
