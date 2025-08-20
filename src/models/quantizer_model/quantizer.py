from torch import nn, einsum
from einops import rearrange, repeat
from torch import autocast

import torch.nn.functional as F

import torch

def exists(
        val: torch.Tensor | None
) -> bool:
    """Check if a tensor exists (is not None).
    """
    return val is not None

def default(
        val: torch.Tensor | None,
        d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the value if it exists, otherwise return the default value.
    """
    return val if exists(val) else d

def log(
        t: torch.Tensor,
        eps: float = 1e-20,
) -> torch.Tensor:
    """Compute the natural logarithm of a tensor with a minimum value (eps) to avoid log(0).
    """
    return torch.log(t.clamp(min = eps))

def uniform_init(
        *shape: tuple[int, ...],
) -> torch.Tensor:
    """Initialize a tensor with a uniform distribution using Kaiming initialization.
    """
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def noop(
        *args,
        **kwargs
) -> None:
    """A no-operation function that does nothing.
    """
    pass

def l2norm(
        t: torch.Tensor,
) -> torch.Tensor:
    """Normalize a tensor along the last dimension using L2 normalization.
    """
    return F.normalize(t, p = 2, dim = -1)

def gumbel_noise(
        t: torch.Tensor,
) -> torch.Tensor:
    """Generate Gumbel noise for the Gumbel-Softmax trick to sample from a categorical distribution.
    """
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(
        t: torch.Tensor,
        temperature: torch.Tensor = 1.,
        dim: int = -1
):
    """Do argmax over a tensor when Gumbel noise has been added to it.

    If temperature is 0, it simply returns the argmax of the tensor along the specified dimension.
    Otherwise, it adds Gumbel noise to the tensor and then performs argmax.

    Args:
        t:
            A tensor from which to sample.
        temperature:
            The temperature parameter by which to scale the tensor 't' before adding Gumbel noise.
        dim:
            The dimension along which to perform the argmax operation.

    Returns:
        A tensor containing the indices of the maximum values along the specified dimension.
    """
    if temperature == 0:
        return t.argmax(dim = dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)


def ema_inplace(
        moving_avg: torch.Tensor,
        new: torch.Tensor,
        decay: float = 0.8,
) -> None:
    """Update the moving average in place using exponential moving average (EMA) formula.
    """
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))
    return

def laplace_smoothing(
        x: torch.Tensor,
        n_categories: int,
        eps: float = 1e-5,
):
    """Apply Laplace smoothing to a tensor.
    """
    return (x + eps) / (x.sum() + n_categories * eps)

def sample_vectors(
        samples: torch.Tensor,
        num: int,
) -> torch.Tensor:
    """Randomly sample a specified number of vectors from a tensor.
    """
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]


def add_noise(
        x: torch.Tensor,
        eps: float =1e-10
) -> torch.Tensor:
    """Add Gaussian noise to a tensor.
    """
    return x + torch.randn_like(x) * eps


def kmeans(
        samples: torch.Tensor,
        num_clusters: int,
        num_iters: int = 10,
        use_cosine_sim: bool = False,
        sample_fn: callable = sample_vectors,
        all_reduce_fn: callable = noop
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform k-means clustering on a set of samples.

    This function initializes cluster means using a sampling function,
    iteratively updates the clusters based on K-means algorithm,
    and returns the final cluster means and the size of each cluster.

    Args:
        samples:
            A tensor of shape (num_samples, dim) representing the data points to be clustered.
        num_clusters:
            The number of clusters to form.
        num_iters:
            The number of iterations to run the K-means algorithm.
        use_cosine_sim:
            If True, use cosine similarity for distance calculation; otherwise, use Euclidean distance.
        sample_fn:
            A function to sample initial cluster means from the samples.
        all_reduce_fn:
            A function to perform all-reduce operation on tensors (useful in distributed settings).

    Returns:
        A tuple (means, bins) where:
            - Means is a tensor of shape (num_clusters, dim) representing the final cluster means.
            - Bins is a tensor of shape (num_clusters,) representing the size of each cluster.
    """
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim = -1)

        buckets = torch.argmax(dists, dim = -1)
        bins = torch.bincount(buckets, minlength = num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins

def orthgonal_loss_fn(
        t: torch.Tensor,
) -> torch.Tensor:
    """Compute the orthogonal loss for a tensor of vectors."""
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = torch.eye(n, device = t.device)
    cosine_sim = einsum('i d, j d -> i j', normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)

class EuclideanCodebook(
    nn.Module
):
    """A codebook for vector quantization where the distances are computed using the Euclidean norm.

    This codebook supports k-means initialization, exponential moving average (EMA) updates,
    and dead code replacement strategies.

    Attributes:
        dim:
            The dimensionality of the vectors in the codebook.
        codebook_size:
            The number of vectors in the codebook.
        kmeans_init:
            If True, initializes the codebook using k-means clustering.
        kmeans_iters:
            The number of iterations for k-means initialization.
        decay:
            The decay factor for EMA updates.
        eps:
            A small value to avoid division by zero in normalization.
        threshold_ema_dead_code:
            The threshold below which a code is considered dead and can be replaced.
        code_replacement_policy:
            The strategy for replacing dead codes ('batch_random' or 'linde_buzo_gray').
        learnable_codebook:
            If True, the codebook is a learnable parameter; otherwise, it is a fixed buffer.
        sample_codebook_temp:
            Temperature for sampling from the codebook during training.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.8,
        eps: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        code_replacement_policy: str = 'batch_random', # batch_random or linde_buzo_gray
        learnable_codebook: bool = False,
        sample_codebook_temp: float = 0
    ) -> None:
        """Initialize the Euclidean codebook for vector quantization.

        This constructor initializes the codebook with the specified parameters like
        dimensionality, size, and initialization method. It also sets up buffers for
        exponential moving averages and cluster sizes. The buffer for the codebook can be
        either a learnable parameter or a fixed buffer depending on the `learnable_codebook` flag.

        Args:
            dim:
                The dimensionality of the vectors in the codebook.
            codebook_size:
                The number of vectors in the codebook.
            kmeans_init:
                If True, initializes the codebook using k-means clustering.
            kmeans_iters:
                The number of iterations for k-means initialization.
            decay:
                The decay factor for exponential moving average updates.
            eps:
                A small value to avoid division by zero in normalization.
            threshold_ema_dead_code:
                The threshold below which a code is considered dead and can be replaced.
            code_replacement_policy:
                The strategy for replacing dead codes ('batch_random' or 'linde_buzo_gray').
            learnable_codebook:
                If True, the codebook is a learnable parameter; otherwise, it is a fixed buffer.
            sample_codebook_temp:
                Temperature for sampling from the codebook during training.
        """
        super().__init__()
        self.decay = decay
        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.code_replacement_policy = code_replacement_policy
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors
        self.all_reduce_fn = noop
        self.add_noise_fn = add_noise

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)
        return

    @torch.jit.ignore
    def init_embed_(
            self,
            data: torch.Tensor,
    ) -> None:
        """Initialize the codebook embeddings using k-means clustering if not already initialized.
        """
        if self.initted:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters,
                                     sample_fn = self.sample_fn, all_reduce_fn = self.all_reduce_fn)


        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
        return

    def replace_batch_random(
            self,
            samples: torch.Tensor,
            mask: torch.Tensor,
    ) -> None:
        """Replace dead codes in the codebook with random samples from the batch.
        """
        samples = l2norm(samples)
        self.embed.data[mask] = self.sample_fn(samples, mask.sum().item())
        return

    def replace_linde_buzo_gray(
            self,
            mask: torch.Tensor,
    ) -> None:
        """Replace dead codes in the codebook using the Linde-Buzo-Gray algorithm.
        """
        num_unused = mask.sum()
        most_used_idxs = self.cluster_size.argsort(descending=True)[:num_unused]
        most_used_codes = self.embed.data[most_used_idxs]
        self.embed.data[mask] = l2norm(self.add_noise_fn(most_used_codes))
        return

    def expire_codes_(
            self,
            batch_samples: torch.Tensor,
    ) -> None:
        """Expire dead codes based on the threshold and replace them using the specified policy.

        This method checks if any codes in the codebook have a cluster size below the specified threshold.
        If such codes are found, it replaces them using the defined code replacement policy.

        Args:
            batch_samples:
                A tensor containing the batch samples which may be used for replacing dead codes when
                the `code_replacement_policy` is set to 'batch_random'.
        """
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        if self.code_replacement_policy == 'batch_random':
            # Replace dead codes by random latents from encoder
            batch_samples = rearrange(batch_samples, '... d -> (...) d')
            self.replace_batch_random(batch_samples, mask = expired_codes)
        elif self.code_replacement_policy == 'linde_buzo_gray':
            # Replace dead codes by most used codes + some noise (Linde-Buzo-Gray splitting algorithm)
            self.replace_linde_buzo_gray(mask = expired_codes)
        else:
            raise ValueError(f'{self.code_replacement_policy} is not a valid dead code replacement strategy.')
        return

    @autocast(device_type='cuda', enabled=False)
    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the codebook to quantize the input tensor.

        This method computes the distances between the input tensor and the codebook embeddings,
        samples from the codebook using Gumbel-Softmax sampling when temperature is greater than zero,
        and updates the codebook embeddings using exponential moving averages (EMA) if in training mode.
        It also computes the perplexity of the sampled embeddings.

        Args:
            x:
                A tensor of shape (..., d) where '...' can be any number of dimensions and 'd' is the dimensionality
                of the input vectors to be quantized.

        Returns:
            A tuple (quantize, embed_ind, perplexity) where:
                - quantize is the quantized representation of the input tensor.
                - embed_ind is the indices of the selected embeddings from the codebook. It has the same shape as the input tensor.
                - perplexity is a measure of how many unique codes are used in the quantization process.
        """
        x = x.float() # if x is a f_map it will look like (b, (hxw), c)

        shape, dtype = x.shape, x.dtype

        flatten = rearrange(x, '... d -> (...) d')

        self.init_embed_(flatten)

        embed = self.embed.t()

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = gumbel_sample(dist, dim = -1, temperature = self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)

        embed_ind = embed_ind.reshape(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            cluster_size = embed_onehot.sum(0)
            self.all_reduce_fn(cluster_size)

            ema_inplace(self.cluster_size, cluster_size, self.decay)

            embed_sum = flatten.t() @ embed_onehot
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        probs = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps)
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs)))
        return {
            'quantize': quantize,
            'embed_ind': embed_ind,
            'perplexity': perplexity
                }



class VectorQuantize(
    nn.Module
):
    """Vector Quantization module that uses a Euclidean codebook for quantization.

    This module supports multi-headed quantization, commitment loss, and orthogonal regularization.
    It can be used to quantize latent representations in models like VQ-VAE.

    Attributes:
        dim:
            The dimensionality of the input vectors.
        codebook_size:
            The number of vectors in the codebook.
        codebook_dim:
            The dimensionality of each vector in the codebook.
        heads:
            The number of heads for multi-headed quantization.
        decay:
            The decay factor for exponential moving average updates.
        eps:
            A small value to avoid division by zero in normalization.
        kmeans_init:
            If True, initializes the codebook using k-means clustering.
        kmeans_iters:
            The number of iterations for k-means initialization.
        threshold_ema_dead_code:
            The threshold below which a code is considered dead and can be replaced.
        code_replacement_policy:
            The strategy for replacing dead codes ('batch_random' or 'linde_buzo_gray', default is 'batch_random').
        accept_image_fmap:
            If True, accepts image feature maps as input.
        commitment_weight:
            The weight for the commitment loss.
        orthogonal_reg_weight:
            The weight for the orthogonal regularization loss.
        orthogonal_reg_active_codes_only:
            If True, applies orthogonal regularization only to active codes.
        orthogonal_reg_max_codes:
            Maximum number of codes to consider for orthogonal regularization.
        sample_codebook_temp:
            Temperature for sampling from the codebook during training.
        norm_latents:
            If True, normalizes encoder latents before quantization.
        weight_loss:
            If True, adds commitment and orthogonal losses to the total loss when these are bigger
            than 0, respectively.
    """
    def __init__(
            self,
            dim: int,
            codebook_size: int,
            codebook_dim: int = None,
            heads: int = 1,
            decay: float = 0.8,
            eps: float = 1e-5,
            kmeans_init: bool = False,
            kmeans_iters: int = 10,
            threshold_ema_dead_code: int = 0,
            code_replacement_policy: str = 'batch_random',
            accept_image_fmap: bool = True,
            commitment_weight: float = 1.,
            orthogonal_reg_weight: float = 0.,
            orthogonal_reg_active_codes_only: bool = False,
            orthogonal_reg_max_codes: bool = None,
            sample_codebook_temp: float = 0.,
            norm_latents: bool = False,
            weight_loss: bool = True,
    ) -> None:
        """Initialize the VectorQuantize module.

        This constructor initializes the vector quantization module with the specified parameters,
        including parameters for the codebook, commitment loss, orthogonal regularization,
        and whether to accept image feature maps as input. It also sets up the necessary layers
        for projecting the input and output of the quantization process. Finally, it initializes the codebook
        using the EuclideanCodebook class.

        Args:
            dim:
                The dimensionality of the input vectors.
            codebook_size:
                The number of vectors in the codebook.
            codebook_dim:
                The dimensionality of each vector in the codebook. If None, defaults to `dim`.
            heads:
                The number of heads for multi-headed quantization.
            decay:
                The decay factor for exponential moving average updates.
            eps:
                A small value to avoid division by zero in normalization.
            kmeans_init:
                If True, initializes the codebook using k-means clustering.
            kmeans_iters:
                The number of iterations for k-means initialization.
            threshold_ema_dead_code:
                The threshold below which a code is considered dead and can be replaced.
            code_replacement_policy:
                The strategy for replacing dead codes ('batch_random' or 'linde_buzo_gray').
            accept_image_fmap:
                If True, accepts image feature maps as input.
            commitment_weight:
                The weight for the commitment loss.
            orthogonal_reg_weight:
                The weight for the orthogonal regularization loss.
            orthogonal_reg_active_codes_only:
                If True, applies orthogonal regularization only to active codes.
            orthogonal_reg_max_codes:
                Maximum number of codes to consider for orthogonal regularization.
            sample_codebook_temp:
                Temperature for sampling from the codebook during training.
            norm_latents:
                If True, normalizes encoder latents before quantization.
            weight_loss:
                If True, adds commitment and orthogonal losses to the total loss when these are bigger
                than 0, respectively.
        """
        super().__init__()
        self.heads = heads
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.commitment_weight = commitment_weight
        self.norm_latents = norm_latents

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        self._codebook = EuclideanCodebook(
            dim=codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            code_replacement_policy=code_replacement_policy,
            learnable_codebook=has_codebook_orthogonal_loss,
            sample_codebook_temp=sample_codebook_temp
        )

        self.codebook_size = codebook_size
        self.accept_image_fmap = accept_image_fmap

        self.weight_loss = weight_loss

    @property
    def codebook(
            self,
    ) -> torch.Tensor:
        """Get the codebook embeddings.
        """
        return self._codebook.embed

    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VectorQuantize module.

        This method quantizes the input tensor `x` using the codebook and computes a series
        of losses including commitment loss and orthogonal regularization loss if specified.
        Note the ability to accept image feature maps as input, which are reshaped accordingly.

        Args:
            x:
                A tensor of shape (batch_size, ..., dim) where '...' can be any number of dimensions.
                If `accept_image_fmap` is True, the expected shape is (batch_size, channels, height, width).

        Returns:
            A tuple (quantize, loss, commit_loss, orthogonal_reg_loss, embed_ind, perplexity) where:
                - quantize is the quantized representation of the input tensor.
                - loss is the total loss combining commitment and orthogonal regularization losses.
                - commit_loss is the commitment loss computed during quantization.
                - orthogonal_reg_loss is the orthogonal regularization loss computed from the codebook. It reflects
                the orthogonality of the codebook vectors.
                - embed_ind is the indices of the selected embeddings from the codebook.
                - perplexity is a measure of how many unique codes are used in the quantization process.
        """
        shape, device, heads, is_multiheaded, codebook_size = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        x = self.project_in(x)

        if self.norm_latents:
            # If specified, normalize encoder latents for computing commitment loss
            x = l2norm(x)

        cb_output = self._codebook(x)

        if self.training:
            cb_output['quantize'] = x + (cb_output['quantize'] - x).detach()

        loss = torch.tensor([0.], device=device, requires_grad=self.training)
        commit_loss = torch.tensor([0.], device=device, requires_grad=self.training)
        orthogonal_reg_loss = torch.tensor([0.], device=device, requires_grad=self.training)

        if self.weight_loss:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(cb_output['quantize'].detach(), x) * self.commitment_weight
                loss = loss + commit_loss

            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook

                if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the activated codes for this batch
                    unique_code_ids = torch.unique(cb_output['embed_ind'])
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthgonal_loss_fn(codebook) * self.orthogonal_reg_weight
                loss = loss + orthogonal_reg_loss

        cb_output['quantize'] = self.project_out(cb_output['quantize'])

        if self.accept_image_fmap:
            cb_output['quantize'] = rearrange(cb_output['quantize'], 'b (h w) c -> b c h w', h=height, w=width)
            cb_output['embed_ind'] = rearrange(cb_output['embed_ind'], 'b (h w) ... -> b h w ...', h=height, w=width)
            if is_multiheaded:
                cb_output['embed_ind'] = rearrange(cb_output['embed_ind'], 'b h w ... -> b ... h w')

        return {
            'quantize': cb_output['quantize'],
            'embed_ind': cb_output['embed_ind'],
            'perplexity': cb_output['perplexity'],
            'loss': loss,
            'commit_loss': commit_loss,
            'orthogonal_reg_loss': orthogonal_reg_loss,
        }



