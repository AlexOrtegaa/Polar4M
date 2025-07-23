import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def build_1d_sincos_posemb(max_len, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings from MoCo-v3, adapted back to 1d

    Returns positional embedding of shape (1, N, D)
    """
    arange = torch.arange(max_len, dtype=torch.float32) # Shape (N,)
    assert embed_dim % 2 == 0, 'Embed dimension must be divisible by 2 for 1D sin-cos position embedding'
    pos_dim = embed_dim // 2
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim # Shape (D/2,)
    omega = 1. / (temperature ** omega)
    out = torch.einsum('n,d->nd', [arange, omega]) # Outer product, shape (N, D/2)
    pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1).unsqueeze(0) # Shape (1, N, D)
    return pos_emb

def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.0):
    """Sine-cosine positional embeddings as used in MoCo-v3

    Returns positional embedding of shape (1, N, D) where N = W*H
    """
    grid_w = torch.arange(w, dtype=torch.float32) # Shape (W,)
    grid_h = torch.arange(h, dtype=torch.float32) # Shape (H, )
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij') # Shapes (W, H)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim # Shape (D/4,)
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('n,d->nd', [grid_w.reshape(-1), omega]) # Outer product, shape (W*H, D/4)
    out_h = torch.einsum('n,d->nd', [grid_h.reshape(-1), omega]) # Outer product, shape (W*H, D/4)
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1).unsqueeze(0) # Shape (1, W*H, D)
    return pos_emb
