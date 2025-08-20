import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VectorQuantize

import torch



class VQVAE(
    nn.Module
):
    """Vector Quantized Variational Autoencoder (VQ-VAE) model.

    This model implements a VQ-VAE architecture that encodes input data into a latent space,
    quantizes the latent representations, and then decodes them back to reconstruct the original input.
    The model consists of an encoder, a vector quantization layer, and a decoder.

    Attributes:
        encoder:
            An instance of the Encoder class that encodes input data into latent representations.
        vector_quantizer:
            An instance of the VectorQuantize class that quantizes the latent representations.
        decoder:
            An instance of the Decoder class that reconstructs the input data from the quantized representations.
    """
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            residual_channels: int,
            num_residual_layers: int,
            codebook_size: int,
            codebook_dim: int,
            commitment_weight: float,
            orthogonal_reg_weight: float,
            sample_codebook_temp: float,

    ):
        """Initialize the VQ-VAE model with the specified parameters.

        Initializes the VQ-VAE model with the given number of input channels, hidden channels,
        residual channels, number of residual layers, codebook size, codebook dimension,
        commitment weight, orthogonal regularization weight, and sample codebook temperature.

        Args:
            in_channels:
                The number of input channels to the encoder.
                It also becomes the number of output channels from the decoder.
            hidden_channels:
                The number of hidden channels in the encoder and decoder.
            residual_channels:
                The number of channels in the residual blocks.
            num_residual_layers:
                The number of residual layers in the encoder and decoder.
            codebook_size:
                The size of the codebook used for vector quantization.
            codebook_dim:
                The dimension of each code in the codebook.
            commitment_weight:
                The weight for the commitment loss in vector quantization.
            orthogonal_reg_weight:
                The weight for the orthogonal regularization loss.
            sample_codebook_temp:
                Temperature parameter for sampling from the codebook.
        """
        super().__init__()

        # encoder
        self.encoder = Encoder(
            in_channels=in_channels, # 1
            out_channels=hidden_channels, # 64
            residual_channels=residual_channels, # 32x
            num_residual_layers=num_residual_layers,
        ) # 64 channels and half of the spatial space

        self.vector_quantization = VectorQuantize(
            dim=hidden_channels,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            heads=1,
            decay=0.8,
            eps=1e-5,
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=0,
            code_replacement_policy='batch_random',  # batch_random or linde_buzo_gray
            commitment_weight=commitment_weight,
            orthogonal_reg_weight=orthogonal_reg_weight,
            orthogonal_reg_active_codes_only=False,
            orthogonal_reg_max_codes=None,
            sample_codebook_temp=sample_codebook_temp,
            norm_latents=True,
        )

        self.decoder = Decoder(
            in_channels=hidden_channels,
            out_channels=in_channels,
            residual_channels=residual_channels,
            num_residual_layers=num_residual_layers,
        )

    def forward(
            self,
            x: torch.Tensor,
    ):
        """Forward pass of the VQ-VAE model.

        This method takes the input tensor, encodes it using the encoder, applies vector quantization,
        and then decodes the quantized representation to reconstruct the original input. It also computes
        various losses associated with the VQ-VAE model.

        Args:
            x:
                A tensor containing the input data to the VQ-VAE model.
                The shape of the tensor should be (batch_size, in_channels, height, width).
        Returns:
            A tuple (quantize, x_hat, loss, recon_loss, vq_loss, commit_loss, orthogonal_reg_loss, embed_ind, perplexity) where:
                - quantize is the quantized latent representation
                - x_hat is the reconstructed output from the decoder
                - loss is the total loss combining reconstruction and VQ losses
                - recon_loss is the reconstruction loss
                - vq_loss is the vector quantization loss
                - commit_loss is the commitment loss
                - orthogonal_reg_loss is the orthogonal regularization loss
                - embed_ind is the indices of the quantized embeddings
                - perplexity is the perplexity of the quantized embeddings
        """
        z = self.encoder(x)

        vq_output = self.vector_quantization(z)
        vq_loss = vq_output['loss']

        x_hat = self.decoder(vq_output['quantize'])

        recon_loss = F.mse_loss(x_hat, x)

        loss = recon_loss + vq_loss

        return {
            'quantize': vq_output['quantize'],
            'x_hat': x_hat,
            'loss': loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'commit_loss': vq_output['commit_loss'],
            'orthogonal_reg_loss': vq_output['orthogonal_reg_loss'],
            'embed_ind': vq_output['embed_ind'],
            'perplexity': vq_output['perplexity'],
        }
