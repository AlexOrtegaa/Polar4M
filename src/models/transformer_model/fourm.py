import torch.nn as nn

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