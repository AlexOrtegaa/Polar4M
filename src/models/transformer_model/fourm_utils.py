from torch import nn

import torch.nn.functional as F

import torch



def softmax1(tensor):
    return F.pad(tensor, (0,1)).softmax(dim=-1)[..., :-1]

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            proj_bias=True,
            attn_drop=0.,
            proj_drop=0.,
            allow_zero_attn=False
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.allow_zero_attn = allow_zero_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #  (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.trasnpose(-2, -1))*self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask, -torch.finfo(attn.dtype).max)

        if self.allow_zero_attn:
            attn = softmax1(attn)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x=self.proj(x)
        x=self.proj_drop(x)
        return x

class NormAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            proj_bias=True,
            norm_layer=nn.LayerNorm,
            attn_drop=0.,
            proj_drop=0.,
            allow_zero_attn=False
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.allow_zero_attn = allow_zero_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim)
        self.k_norm = norm_layer(head_dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #  (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.trasnpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask, -torch.finfo(attn.dtype).max)

        if self.allow_zero_attn:
            attn = softmax1(attn)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias=False,
            proj_bias=True,
            mlp_bias=False,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            gated_mlp=False,
            qkv_norm=False,
            allow_zero_attn=False,
         ):
        super().__init__()
        self.norm1=norm_layer(dim)

        if not qkv_norm:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                allow_zero_attn=allow_zero_attn,
            )
        else:
            self.attn = NormAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                proj_drop=drop,
                allow_zero_attn=allow_zero_attn,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if not gated_mlp:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                bias=mlp_bias,
                drop=drop,
            )
        else:
            self.mlp = GatedMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                bias=mlp_bias,
            )

        def forward(self, x, mask=None):
            x =  x+self.drop_path(self.attn(self.norm1(x), mask=mask))
            x = x+self.drop_path(self.mlp(self.norm2(x)))
            return x