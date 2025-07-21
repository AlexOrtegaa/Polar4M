from torch import nn

import torch.nn.functional as F

import torch



class VectorQuantizer(nn.Module):
    def __init__(self, beta, num_embeddings, dim_embeddings):
        super().__init__()
        self.commitment_loss = beta
        self.num_embeddings = int(num_embeddings)
        self.dim_embeddings = int(dim_embeddings)

        self.embedding = nn.Embedding(self.num_embeddings, self.dim_embeddings)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0/self.num_embeddings)

    def forward(self, z):
        z =  z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.dim_embeddings)

        distances = torch.sum(z_flat ** 2, dim=1, keepdim=True) +\
        torch.sum(self.embedding.weight ** 2, dim=1) -\
            2*torch.matmul(z_flat, self.embedding.weight.t())

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(z.dtype)

        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(z.shape)

        loss = F.mse_loss(quantized.detach(), z) + self.commitment_loss * F.mse_loss(quantized, z.detach())

        quantized = z + (quantized -z).detach()

        probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))

        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return loss, quantized, perplexity, encodings, encoding_indices


