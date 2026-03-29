import math
import numpy as np
import torch
from torch import nn


class LlamaEntropyStableRotaryEmbedding(nn.Module):
    """Entropy-Stable RoPE - Attention scaling based on entropy stability.

    This module implements a rotary position embedding with entropy-stable scaling
    to maintain attention distribution stability when extending beyond training length.

    Attributes:
        dim (int): Attention head dimension.
        base (int): RoPE base frequency, default is 10000.
        max_position_embeddings (int): Maximum position embedding length.
        original_max_position_embeddings (int): Original training length L_0.
        scaling_factor (float): Static scaling ratio S.
        dynamic (bool): Whether to compute scaling dynamically.
        alpha (float): Base scaling coefficient (default 0.1, same as YaRN).
        beta (float): Entropy compensation strength (default 1.0).
        inv_freq (torch.Tensor): Precomputed inverse frequency buffer.
        cos_cached (torch.Tensor): Cached cosine values for rotary embeddings.
        sin_cached (torch.Tensor): Cached sine values for rotary embeddings.
        compensation_factors (torch.Tensor): Precomputed entropy compensation factors.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device=None,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 2048,
        dynamic: bool = False,
        alpha: float = 0.1,
        beta: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scaling_factor = scaling_factor
        self.dynamic = dynamic
        self.alpha = alpha
        self.beta = beta

        # Precompute base parameters
        self._precompute_params(device)

        # Precompute entropy compensation factors
        self._precompute_compensation_factors()

        if not dynamic:
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=device,
                dtype=torch.get_default_dtype(),
            )

    def _precompute_params(self, device):
        """
        Precompute base parameters for rotary embeddings.

        Args:
            device: Device to place the computed tensors on.
        """
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _precompute_compensation_factors(self):
        """
        Precompute entropy compensation factors for positions up to 16k.

        This method computes compensation factors using a logarithmic growth model
        for positions beyond the original training length.
        """
        L0 = self.original_max_position_embeddings
        max_pos = 16384  # Precompute up to 16k positions

        factors = []
        for pos in range(max_pos):
            if pos < L0:
                # Within training length: no compensation
                factor = 1.0
            else:
                # Beyond training length: logarithmic growth compensation
                rel_pos = pos - L0
                rel_ratio = rel_pos / L0

                # Relative entropy (logarithmic growth model)
                rel_entropy = 1.0 + 0.5 * math.log(1 + rel_ratio) / math.log(2)

                # Compensation factor
                factor = 1.0 + self.beta * (rel_entropy - 1.0)

                # Limit maximum compensation
                factor = min(factor, 2.5)

            factors.append(factor)

        self.compensation_factors = torch.tensor(factors, dtype=torch.float32)

    def _compute_compensation_factor(self, position: int) -> float:
        """
        Compute entropy compensation factor for a given position.

        Args:
            position (int): The position index to compute compensation for.

        Returns:
            float: The compensation factor for the given position.
        """
        if position < len(self.compensation_factors):
            return self.compensation_factors[position].item()
        else:
            # Beyond precomputed range, use maximum value
            return 2.5

    def _compute_attention_scaling(self, position: int, S: float) -> float:
        """
        Compute attention scaling coefficient.

        The scaling is computed as: t(position, S) = 1 + α · ln(S) · compensation_factor(position)

        Args:
            position (int): The position index.
            S (float): The scaling ratio.

        Returns:
            float: The attention scaling coefficient.
        """
        if S <= 1.0:
            return 1.0

        # Entropy compensation factor
        compensation_factor = self._compute_compensation_factor(position)

        # Combined scaling
        t = 1.0 + self.alpha * math.log(S) * compensation_factor

        return t

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """
        Set up cosine and sine cache for rotary embeddings.

        Args:
            seq_len (int): Sequence length to cache.
            device: Device to place the cached tensors on.
            dtype: Data type for the cached tensors.
        """
        self.max_seq_len_cached = seq_len
        inv_freq = self.inv_freq.to(device=device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Standard RoPE
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Apply entropy-stable scaling
        S = self.scaling_factor
        scaling_factors = torch.tensor(
            [self._compute_attention_scaling(int(pos), S) for pos in range(seq_len)],
            device=device,
            dtype=dtype,
        )

        self.register_buffer(
            "cos_cached",
            emb.cos().to(dtype) * scaling_factors[:, None],
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            emb.sin().to(dtype) * scaling_factors[:, None],
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """
        Forward pass to compute rotary position embeddings.

        Args:
            x (torch.Tensor): Input tensor with shape [batch, num_heads, seq_len, head_dim].
            seq_len (int, optional): Sequence length (used in dynamic mode). Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of cosine and sine components of rotary
                embeddings with entropy-stable scaling applied.
        """
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            # Dynamic mode: compute scaling in real-time
            S = max(1.0, seq_len / self.original_max_position_embeddings)
            inv_freq = self.inv_freq.to(device=device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)

            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            # Dynamically compute entropy-stable scaling
            scaling_factors = torch.tensor(
                [
                    self._compute_attention_scaling(int(pos), S)
                    for pos in range(seq_len)
                ],
                device=device,
                dtype=dtype,
            )

            return (
                emb.cos().to(dtype=dtype) * scaling_factors[:, None],
                emb.sin().to(dtype=dtype) * scaling_factors[:, None],
            )
        else:
            # Static mode: use cache
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=dtype),
                self.sin_cached[:seq_len].to(dtype=dtype),
            )
