import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VectorQuantizer



class VQVAE(nn.Module):

    def __init__(self, in_channels, hidden_channels, residual_channels, num_residual_layers,
                 num_embeddings, dim_embeddings, beta):
        super().__init__()

        # encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=hidden_channels,
            residual_channels=residual_channels,
            num_residual_layers=num_residual_layers,
        )

        self.pre_quantization_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=dim_embeddings,
            kernel_size=1,
            stride=1,
        )

        self.vector_quantization = VectorQuantizer(
            beta=beta,
            num_embeddings=num_embeddings,
            dim_embeddings=dim_embeddings,
        )

        self.post_quantization_conv = nn.ConvTranspose2d(
            in_channels=dim_embeddings,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
        )

        self.decoder = Decoder(
            in_channels=hidden_channels,
            out_channels=in_channels,
            residual_channels=residual_channels,
            num_residual_layers=num_residual_layers,
        )

    def forward(self, x):

        z = self.encoder(x)

        z = self.pre_quantization_conv(z)

        vq_loss, quantized, perplexity, encodings, encoding_indices = self.vector_quantization(z)

        z_quantized = self.post_quantization_conv(quantized)

        x = self.decoder(z_quantized)

        return vq_loss, x, perplexity
