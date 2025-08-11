import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VectorQuantize



class VQVAE(nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_channels,
            residual_channels,
            num_residual_layers,
            codebook_size,
            codebook_dim,
            commitment_weight,
            orthogonal_reg_weight,
            sample_codebook_temp,

    ):
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

    def forward(self, x):

        z = self.encoder(x)

        quantize, vq_loss, commit_loss, orthogonal_reg_loss, embed_ind, perplexity = self.vector_quantization(z)

        x_recon = self.decoder(quantize)

        recon_loss = F.mse_loss(x_recon, x)

        loss = recon_loss + vq_loss

        return (quantize, x_recon,
                loss,
                recon_loss,
                vq_loss, commit_loss, orthogonal_reg_loss,
                embed_ind, perplexity)
