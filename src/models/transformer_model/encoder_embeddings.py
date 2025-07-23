from typing import Dict, Optional, Tuple, Union
from .fm_utils import build_2d_sincos_posemb, pair
from einops import repeat

import torch.nn as nn

import torch



class ImageTokenEncoderEmbedding(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            patch_size: Union[int, Tuple[int,int]] = 16,
            dim_tokens: Optional[int] = None,
            sincos_pos_emb: bool = True,
            image_size: Union[int, Tuple[int]] = 224,
            **kwargs,
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size) * (self.image_size[1] // patch_size)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(
            self,
            dim_tokens: int = 768,
            init_std=0.02
    ):

        self.dim_tokens = dim_tokens

        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]

        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.register_buffer("pos_emb", pos_emb)  # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding
        self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ids = d['tensor']
        B = ids.shape[0]
        ids = ids.reshape(B, -1)

        # Map to embedding
        x = self.token_emb(ids)

        # Create positional embedding + modality embedding
        x_emb = repeat(self.pos_emb + self.mod_emb, '() n d -> b n d', b=B)

        d['x'] = x
        d['emb'] = x_emb

        return d
