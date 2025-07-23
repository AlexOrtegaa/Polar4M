from typing import Dict
from functools import partial

import torch.nn as nn
import torch.nn.functional as F

import torch



class FourM(nn.Module):
    def __init__(self,
                share_modality_embeddings,
                shared_drop_path,
                encoder_embeddings,
                decoder_embeddings,
                drop_path_rate_encoder,
                dim,
                encoder_depth,
                decoder_depth
            ):
        super.__init__()

        self.encoder_modalities = set(encoder_embeddings.keys())
        for embedding in encoder_embeddings.values():
            embedding.init(dim_tokens=dim, init_std=self.init_std)
        self.encoder_embeddings = nn.ModuleDict(encoder_embeddings)

        self.decoder_modalities = set(self.encoder_embeddings.keys())
        for embedding in self.decoder_modalities:
            embedding.init(dim_tokens=dim, init_std=self.init_std)
        self.decoder_embeddings = nn.ModuleDict(decoder_embeddings)

        if share_modality_embeddings:
            self.share_modality_embeddings()

        if shared_drop_path:
            dropout_encoder = [x.item() for x in torch.linspace(0, drop_path_rate_encoder, encoder_depth+decoder_depth)][:encoder_depth]
        else:
            dropout_encoder = [x.item() for x in torch.linspace(0, drop_path_rate_encoder, encoder_depth)]

        self.encoder = nn.ModuleList([
            Block(

            )
            for i in range(encoder_depth)
        ])

    def get_num_layers_encoder(
            self,
    ):
        return len(self.encoder)

    def get_num_layers_decoder(
            self,
    ):
        return len(self.decoder)

    def get_num_layers(
            self,
    ):
        return self.get_num_layers_encoder() + self.get_num_layers_decoder()

    def forwards_encoder(
            self,
            x,
            encoder_mask,
            **kwargs
    ):
        for block in self.encoder:
            x = block(x, mask=encoder_mask)

        x = self.encoder_norm(x)
        return x

    def forward_decoder(
            self,
            y,
            context,
            encoder_mask,
            decoder_attention_mask,

    ):
        for block in self.decoder:
            y = block(y, context, sa_mask=decoder_attention_mask, xa_mask=encoder_mask)

        y = self.decoder_norm(y)
        return y

    def forward_loss(
            self,
            y,
            target_ids,
            decoder_mod_dict,
            decoder_mod_mask,
            loss_type,
    ):
        if loss_type in ['mod', 'modality']:
            loss, mod_loss = self.foward_modality_loss(
                y,
                target_ids,
                decoder_mod_dict,
                decoder_mod_mask,
            )
        elif loss_type == 'token':
            loss, mod_loss = self.forward_token_loss(
                y,
                target_ids,
                decoder_mod_dict,
                decoder_mod_mask,
            )
        else:
            raise ValueError(f"Invalid loss type")

        return loss, mod_loss

    def forward_mod_loss(
            self,
            y,
            target_ids,
            decoder_mod_dict,
            decoder_mod_mask,
    ):
        mod_loss={}
        for mod, d in decoder_mod_dict.items():
            idx = self.modality_info[mod]["id"]
            logits = self.decoder_embeddings[mod].forward_logits(y[decoder_mod_mask == idx])
            if logits.numel() == 0:
                # If there are no logits / targets, set mod_loss to 0
                mod_loss[mod] = torch.zeros(1, device=logits.device)
            else:
                loss = F.cross_entropy(logits, target_ids[decoder_mod_mask == idx].long(), reduction='mean')
                mod_loss[mod] = loss

        loss = sum(mod_loss.values()) / len(mod_loss)

        return loss, mod_loss


### Freezing section
    def freeze_encoder(
            self,
            freeze_embeddings=True
    ):
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder_norm.parameters():
            param.requires_grad = False

        if freeze_embeddings:
            for param in self.encoder_embeddings.parameters():
                param.requires_grad = False

    def freeze_encoder_except_specific_embeddings(
            self,
            frozen_embedding_domain
    ):
        frozen_embedding_domain = frozen_embedding_domain.split('-')
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder_norm.parameters():
            param.requires_grad = False

        for name, param in self.encoder_embeddings.named_parameters():
            if name.split('.')[0] in frozen_embedding_domain:
                param.requires_grad = False

    def unfreeze_encoder(
            self,
            unfreeze_embeddings=True
    ):
        for param in self.encoder.parameters():
            param.requires_grad = True

        for param in self.encoder_norm.parameters():
            param.requires_grad = True

        if unfreeze_embeddings:
            for param in self.encoder_embeddings.parameters():
                param.requires_grad = True

    def freeze_decoder(
            self,
            freeze_embeddings=True
    ):
        for param in self.decoder.parameters():
            param.requires_grad = False

        for param in self.decoder_norm.parameters():
            param.requires_grad = False

        if freeze_embeddings:
            for param in self.decoder_embeddings.parameters():
                param.requires_grad = False

    def freeze_decoder_except_specific_embeddings(
        self,
        frozen_embedding_domain
    ):
        frozen_embedding_domain = frozen_embedding_domain.split('-')
        for param in self.decoder.parameters():
            param.requires_grad = False

        for param in self.decoder_norm.parameters():
            param.requires_grad = False

        for name, param in self.decoder_embeddings.named_parameters():
            if name.split('.')[0] in frozen_embedding_domain:
                param.requires_grad = False

    def unfreeze_decoder(
            self,
            unfreeze_embeddings=True
    ):
        for param in self.decoder.parameters():
            param.requires_grad = True

        for param in self.decoder_norm.parameters():
            param.requires_grad = True

        if unfreeze_embeddings:
            for param in self.decoder_embeddings.parameters():
                param.requires_grad = True

    def freeze_shared_params(
            self
    ):
        self.freeze_encoder(freeze_embeddings=False)
        self.freeze_decoder(freeze_embeddings=False)

    def freeze_params_except_specific_embeddings(
            self,
            frozen_embedding_domain
    ):
        self.freeze_encoder_except_specific_embeddings(frozen_embedding_domain=frozen_embedding_domain)
        self.freeze_decoder_except_specific_embeddings(frozen_embedding_domain=frozen_embedding_domain)

    def unfreeze_shared_params(
            self
    ):
        self.unfreeze_encoder(unfreeze_embeddings=False)
        self.unfreeze_decoder(unfreeze_embeddings=False)

    def unfreeze_all(
            self
    ):
        self.unfreeze_encoder(unfreeze_embeddings=True)
        self.unfreeze_decoder(unfreeze_embeddings=True)


def fm_base_12e_12d_gelu(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=12,
        decoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def fm_small_8e_8d_gelu(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=8,
        decoder_depth=8,
        dim=512,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model