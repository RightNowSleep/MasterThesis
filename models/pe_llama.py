"""Rotary Position Embedding (RoPE) implementations for LLaMA models.

This module provides a comprehensive collection of RoPE scaling methods for extending
the context window of LLaMA models beyond their original training length. It includes
both standard and novel approaches to position encoding with support for static and
dynamic scaling modes.

Supported RoPE Scaling Methods:
    - Standard RoPE: Original rotary position embedding from the LLaMA paper.
    - Position Interpolation (PI): Linear position scaling for context extension.
    - NTK-aware: Base frequency modification approach for better extrapolation.
    - NTK-by-parts: Piecewise frequency blending for high/low frequency dimensions.
    - YaRN: NTK-by-parts combined with attention temperature scaling.
    - My RoPE / My RoPE 2: Layer-aware custom RoPE with adaptive parameters.
    - Block-Layered: Quantized effective positions using exponential block sizes.
    - Freq-Smooth: Quadratic block-size schedule with C1 smoothness.
    - Freq-Reciprocal: Constant-product relationship between block size and frequency decay.
    - Dual RoPE / Inverse Dual RoPE: Novel dual-position encoding approaches.

Static/Dynamic Unification:
    Each implementation class accepts a ``dynamic`` constructor argument:

    - ``dynamic=False`` (default): Static mode where frequencies are pre-cached once
      at construction time. The scaling factor is fixed (= config "factor").
    - ``dynamic=True``: Dynamic mode where the effective scaling factor is recomputed
      on every forward pass as s = max(1, seq_len / original_L). When seq_len <= original_L,
      s=1 and the class degrades to plain RoPE (no overhead).

All implementations follow a consistent interface with forward(x, seq_len) returning
(cos, sin) tuples for rotary position embedding application.
"""

import math

import torch
from torch import nn

__all__ = [
    "LlamaRotaryEmbedding",
    "LlamaLinearScalingRotaryEmbedding",
    "LlamaNTKAwareScaledRotaryEmbedding",
    "LlamaNTKByPartsScaledRotaryEmbedding",
    "LlamaYarnScaledRotaryEmbedding",
    # My RoPE family
    "LlamaMyRotaryEmbedding",
    "LlamaMyScaledRotaryEmbedding",
    # My RoPE 2 family
    "LlamaMyRotaryEmbedding2",
    "LlamaMyScaledRotaryEmbedding2",
    # Block-Layered family
    "LlamaBlockLayeredRotaryEmbedding",
    "LlamaBlockLayeredScaledRotaryEmbedding",
    # Freq-Smooth family
    "LlamaFreqSmoothRotaryEmbedding",
    "LlamaFreqSmoothScaledRotaryEmbedding",
    # Freq-Reciprocal family
    "LlamaFreqReciprocalRotaryEmbedding",
    "LlamaFreqReciprocalScaledRotaryEmbedding",
    "LlamaFreqReciprocalScaledNoLayerRotaryEmbedding",
    "LlamaFreqReciprocalScaledAdaptiveRotaryEmbedding",
    # Dual RoPE family
    "LlamaDualRoPEEmbedding",
    "LlamaDualRoPEScaledEmbedding",
    # Inverse Dual RoPE family
    "LlamaInverseDualRoPEEmbedding",
    "LlamaInverseDualRoPEScaledEmbedding",
]

# ================================================================================== #
#  RoPE implementations                                                              #
#                                                                                    #
#  Static/dynamic unification                                                        #
#  ──────────────────────────                                                        #
#  LlamaLinearScalingRotaryEmbedding      (PI)                                       #
#  LlamaNTKAwareScaledRotaryEmbedding     (NTK-aware)                                #
#  LlamaNTKByPartsScaledRotaryEmbedding   (NTK-by-parts)                             #
#  LlamaYarnScaledRotaryEmbedding         (YaRN)                                     #
#  LlamaMyRotaryEmbedding                 (My RoPE  — layer-aware, position only)    #
#  LlamaMyScaledRotaryEmbedding           (My RoPE  + attention temperature)         #
#  LlamaMyRotaryEmbedding2                (My RoPE2 — multi-scale subspace, pos only)#
#  LlamaMyScaledRotaryEmbedding2          (My RoPE2 + attention temperature)         #
#  LlamaBlockLayeredRotaryEmbedding       (Block-Layered RoPE, position only)        #
#  LlamaBlockLayeredScaledRotaryEmbedding (Block-Layered RoPE + attn temperature)    #
#  LlamaFreqSmoothRotaryEmbedding         (Freq-Smooth RoPE, position only)          #
#  LlamaFreqSmoothScaledRotaryEmbedding   (Freq-Smooth RoPE + attn temperature)      #
#  LlamaFreqReciprocalRotaryEmbedding     (Freq-Reciprocal RoPE, position only)      #
#  LlamaFreqReciprocalScaledRotaryEmbedding (Freq-Reciprocal RoPE + attn temperature)#
#  LlamaFreqReciprocalScaledNoLayerRotaryEmbedding                                   #
#  (Freq-Reciprocal RoPE + attn temperature, no layer index)                         #
#  LlamaFreqReciprocalScaledAdaptiveRotaryEmbedding                                  #
#                                                                                    #
#  Each class accepts a ``dynamic: bool`` constructor argument:                      #
#    dynamic=False (default) – static mode: frequencies are pre-cached once          #
#                              at construction time.  The scaling factor is          #
#                              fixed (= config "factor").                            #
#    dynamic=True            – dynamic mode: the effective scaling factor is         #
#                              recomputed on **every** forward pass as               #
#                              s = max(1, seq_len / original_L).                     #
#                              When seq_len <= original_L, s=1 and the class         #
#                              degrades to plain RoPE (no overhead).                 #
#                                                                                    #
#  My-RoPE variants are kept separate (not merged) by design.                        #
# ================================================================================== #


def _interleave_cos_sin(freqs):
    """Compute cosine and sine values from frequency tensor for rotary embeddings.

    Takes a frequency tensor of shape (seq_len, dim//2) and produces
    cos/sin tensors of shape (seq_len, dim) by duplicating the frequency
    dimensions to match the HuggingFace RoPE convention used by apply_rotary_pos_emb.

    The output layout is [f0, f1, ..., f_{n-1}, f0, f1, ..., f_{n-1}]
    where the first half and second half are identical. This matches the
    expected format in modeling_llama.apply_rotary_pos_emb which computes:
        q_embed = q * cos + rotate_half(q) * sin
    where rotate_half swaps the first/second halves, requiring cos[i] == cos[i+n].

    Args:
        freqs (torch.Tensor): Frequency tensor of shape (seq_len, dim//2).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - cos: Cosine values of shape (seq_len, dim).
            - sin: Sine values of shape (seq_len, dim).
    """
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


class LlamaRotaryEmbedding(nn.Module):
    """Standard Llama Rotary Position Embedding (RoPE).

    Implements the original RoPE as described in the LLaMA paper. Computes
    cos/sin caches for rotary position embeddings with fixed base frequency.
    This is the baseline implementation without any context window extension.

    Attributes:
        dim (int): Dimension of the embedding (head dimension).
        max_position_embeddings (int): Maximum sequence length for caching.
        base (int): Base frequency for computing inverse frequencies.
        inv_freq (torch.Tensor): Inverse frequency buffer of shape (dim//2,).
        cos_cached (torch.Tensor): Cached cosine values of shape (max_seq_len, dim).
        sin_cached (torch.Tensor): Cached sine values of shape (max_seq_len, dim).
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """Initialize the standard Rotary Position Embedding.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency for computing inverse frequencies.
                Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """Compute and cache cosine and sine values for rotary embeddings.

        Precomputes the cos/sin tables for all positions up to seq_len using
        the standard RoPE formula with fixed inverse frequencies.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached,
            device=device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = _interleave_cos_sin(freqs)
        self.register_buffer("cos_cached", cos.contiguous().to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.contiguous().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """Forward pass to retrieve cached or compute new rotary embeddings.

        Returns cached cos/sin values for the requested sequence length.
        If seq_len exceeds cache size, automatically recomputes the cache.

        Args:
            x (torch.Tensor): Input tensor used to determine device and dtype.
                Shape is (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from x.shape[2]. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Cosine values of shape (seq_len, dim).
                - sin: Sine values of shape (seq_len, dim).
        """
        if seq_len is None:
            seq_len = x.shape[2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# ---------------------------------------------------------------------------- #
#  PI  (Position Interpolation / Linear scaling)                               #
# ---------------------------------------------------------------------------- #


class LlamaLinearScalingRotaryEmbedding(nn.Module):
    """Position Interpolation (PI) with linear position scaling.

    Implements linear position scaling where positions are divided by the
    scaling factor to extend the context window. This is the simplest form
    of RoPE scaling, directly interpolating positions into a larger range.

    Supports both static mode (fixed scaling factor) and dynamic mode
    (scaling factor computed at runtime based on sequence length).

    Attributes:
        dim (int): Dimension of the embedding.
        max_position_embeddings (int): Maximum sequence length for caching.
        original_max_position_embeddings (int): Model's original context length.
        base (int): Base frequency for computing inverse frequencies.
        scaling_factor (float): Extension ratio for static mode.
        dynamic (bool): Whether to use dynamic scaling.
        inv_freq (torch.Tensor): Inverse frequency buffer.
        cos_cached (torch.Tensor): Cached cosine values (static mode only).
        sin_cached (torch.Tensor): Cached sine values (static mode only).
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=2048,
        dynamic=False,
    ):
        """Initialize linear scaled rotary position embedding.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency for computing inverse frequencies.
                Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Extension ratio for static mode (>1.0 extends
                context window). Defaults to 1.0.
            original_max_position_embeddings (int): Model's original training
                context length. Defaults to 2048.
            dynamic (bool): Whether to use dynamic scaling. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.dynamic = dynamic

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32).to(device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._dynamic_seq_len_cached = -1
        self._dynamic_cos_cached = None
        self._dynamic_sin_cached = None

        if not dynamic:
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=self.inv_freq.device,
                dtype=torch.get_default_dtype(),
            )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """Compute and cache cos/sin values with linear position scaling.

        Applies linear interpolation by dividing position indices by the
        scaling factor before computing frequencies.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = _interleave_cos_sin(freqs)
        self.register_buffer("cos_cached", cos.contiguous().to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.contiguous().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """Forward pass to retrieve or compute linearly-scaled rotary embeddings.

        In static mode, uses cached values. In dynamic mode, computes scaling
        factor as max(seq_len, original_max) / original_max.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Scaled cosine values of shape (seq_len, dim).
                - sin: Scaled sine values of shape (seq_len, dim).
        """
        if seq_len is None:
            seq_len = x.shape[2]
        if self.dynamic:
            s = (
                max(seq_len, self.original_max_position_embeddings)
                / self.original_max_position_embeddings
            )
            if seq_len == self._dynamic_seq_len_cached:
                return (
                    self._dynamic_cos_cached.to(dtype=x.dtype),
                    self._dynamic_sin_cached.to(dtype=x.dtype),
                )
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            t = t / s
            freqs = torch.outer(t, self.inv_freq)
            cos, sin = _interleave_cos_sin(freqs)
            cos = cos.to(dtype=x.dtype)
            sin = sin.to(dtype=x.dtype)
            self._dynamic_seq_len_cached = seq_len
            self._dynamic_cos_cached = cos.detach()
            self._dynamic_sin_cached = sin.detach()
            return cos, sin
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=x.dtype),
                self.sin_cached[:seq_len].to(dtype=x.dtype),
            )


# ---------------------------------------------------------------------------- #
#  NTK-aware scaling                                                           #
# ---------------------------------------------------------------------------- #


class LlamaNTKAwareScaledRotaryEmbedding(nn.Module):
    """NTK-aware RoPE scaling with base frequency modification.

    Modifies the RoPE base frequency to extend context length while preserving
    high-frequency information better than simple linear interpolation. The base
    is scaled as base' = base * s^(d/(d-2)) where s is the scaling factor and d
    is the dimension.

    Supports both static and dynamic scaling modes.

    Attributes:
        dim (int): Dimension of the embedding.
        max_position_embeddings (int): Maximum sequence length for caching.
        original_max_position_embeddings (int): Model's original context length.
        base (int): Original base frequency.
        scaling_factor (float): Extension ratio for static mode.
        dynamic (bool): Whether to use dynamic scaling.
        _exp_factor (float): Exponent factor computed as dim / (dim - 2).
        inv_freq (torch.Tensor): Inverse frequency buffer.
        cos_cached (torch.Tensor): Cached cosine values (static mode only).
        sin_cached (torch.Tensor): Cached sine values (static mode only).
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=2048,
        dynamic=False,
    ):
        """Initialize NTK-aware scaled rotary position embedding.

        Args:
            dim (int): Dimension of the embedding (must be even, >2).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Original base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Extension ratio for static mode. Defaults to 1.0.
            original_max_position_embeddings (int): Model's original context length.
                Defaults to 2048.
            dynamic (bool): Whether to use dynamic scaling. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.dynamic = dynamic
        self._exp_factor = dim / (dim - 2)

        self._dynamic_seq_len_cached = -1
        self._dynamic_s_cached = -1.0
        self._dynamic_cos_cached = None
        self._dynamic_sin_cached = None

        if not dynamic:
            modified_base = base * scaling_factor**self._exp_factor
            inv_freq = 1.0 / (
                modified_base
                ** (torch.arange(0, dim, 2, dtype=torch.float32).to(device) / dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=self.inv_freq.device,
                dtype=torch.get_default_dtype(),
            )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """Compute and cache cos/sin values with NTK-aware modified base frequency.

        Uses the modified base frequency that preserves high-frequency information
        better than linear interpolation.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = _interleave_cos_sin(freqs)
        self.register_buffer("cos_cached", cos.contiguous().to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.contiguous().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """Forward pass to retrieve or compute NTK-aware scaled rotary embeddings.

        In dynamic mode, modifies the base frequency based on current sequence
        length relative to original context length.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: NTK-scaled cosine values of shape (seq_len, dim).
                - sin: NTK-scaled sine values of shape (seq_len, dim).
        """
        if seq_len is None:
            seq_len = x.shape[2]
        if self.dynamic:
            s = max(1.0, seq_len / self.original_max_position_embeddings)
            if seq_len == self._dynamic_seq_len_cached and s == self._dynamic_s_cached:
                return (
                    self._dynamic_cos_cached.to(dtype=x.dtype),
                    self._dynamic_sin_cached.to(dtype=x.dtype),
                )
            modified_base = self.base * s**self._exp_factor
            inv_freq = 1.0 / (
                modified_base
                ** (
                    torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32)
                    / self.dim
                )
            )
            t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            cos, sin = _interleave_cos_sin(freqs)
            cos = cos.to(dtype=x.dtype)
            sin = sin.to(dtype=x.dtype)
            self._dynamic_seq_len_cached = seq_len
            self._dynamic_s_cached = s
            self._dynamic_cos_cached = cos.detach()
            self._dynamic_sin_cached = sin.detach()
            return cos, sin
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=x.dtype),
                self.sin_cached[:seq_len].to(dtype=x.dtype),
            )


# ---------------------------------------------------------------------------- #
#  NTK-by-parts scaling                                                        #
# ---------------------------------------------------------------------------- #


class LlamaNTKByPartsScaledRotaryEmbedding(nn.Module):
    """NTK-by-parts RoPE scaling with piecewise frequency blending.

    Splits dimensions into three regions based on rotation frequency within
    the original context window:
    - High-frequency dimensions (r >= beta): unchanged, preserve fine details.
    - Low-frequency dimensions (r <= alpha): linearly interpolated.
    - Transition region (alpha < r < beta): smooth blend between strategies.

    This hybrid approach aims to combine the benefits of both NTK-aware scaling
    for low frequencies and preservation of high-frequency information.

    Attributes:
        dim (int): Dimension of the embedding.
        max_position_embeddings (int): Maximum sequence length for caching.
        original_max_position_embeddings (int): Model's original context length.
        base (int): Base frequency for computing inverse frequencies.
        scaling_factor (float): Extension ratio for static mode.
        alpha_ntk (float): Lower boundary for transition region (default 1.0).
        beta_ntk (float): Upper boundary for transition region (default 32.0).
        dynamic (bool): Whether to use dynamic scaling.
        inv_freq (torch.Tensor): Inverse frequency buffer.
        cos_cached (torch.Tensor): Cached cosine values (static mode only).
        sin_cached (torch.Tensor): Cached sine values (static mode only).
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=2048,
        alpha=1.0,
        beta=32.0,
        dynamic=False,
    ):
        """Initialize NTK-by-parts scaled rotary position embedding.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Extension ratio for static mode. Defaults to 1.0.
            original_max_position_embeddings (int): Model's original context length.
                Defaults to 2048.
            alpha (float): Lower boundary for frequency blending. Defaults to 1.0.
            beta (float): Upper boundary for frequency blending. Defaults to 32.0.
            dynamic (bool): Whether to use dynamic scaling. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.alpha_ntk = alpha
        self.beta_ntk = beta
        self.dynamic = dynamic

        theta_d = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )
        self.register_buffer("_theta_d_base", theta_d, persistent=False)

        lambda_d = 2.0 * math.pi / theta_d
        r_d = self.original_max_position_embeddings / lambda_d
        w_ext_base = torch.clamp(
            (r_d - self.alpha_ntk) / (self.beta_ntk - self.alpha_ntk),
            0.0,
            1.0,
        )
        self.register_buffer("_w_ext_base", w_ext_base, persistent=False)

        self._dynamic_seq_len_cached = -1
        self._dynamic_s_cached = -1.0
        self._dynamic_cos_cached = None
        self._dynamic_sin_cached = None

        if not dynamic:
            inv_freq = self._compute_inv_freq(scaling_factor)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=self.inv_freq.device,
                dtype=torch.get_default_dtype(),
            )

    def _compute_inv_freq(self, scaling_factor):
        """Compute blended inverse frequencies for NTK-by-parts scaling.

        Combines original and interpolated inverse frequencies using
        dimension-dependent blending weights.

        Args:
            scaling_factor (float): The scaling factor to apply.

        Returns:
            torch.Tensor: Blended inverse frequencies of shape (dim//2,).
        """
        theta_d = self._theta_d_base
        w_ext = self._w_ext_base
        return theta_d * w_ext + (1.0 - w_ext) * theta_d / scaling_factor

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """Compute and cache cos/sin values with NTK-by-parts blending.

        Uses piecewise frequency blending to compute embeddings that preserve
        high frequencies while extending low-frequency context.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = _interleave_cos_sin(freqs)
        self.register_buffer("cos_cached", cos.contiguous().to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.contiguous().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """Forward pass to retrieve or compute NTK-by-parts scaled rotary embeddings.

        Applies piecewise frequency blending based on dimension-specific
        rotation counts within the original context window.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Blended cosine values of shape (seq_len, dim).
                - sin: Blended sine values of shape (seq_len, dim).
        """
        if seq_len is None:
            seq_len = x.shape[2]
        if self.dynamic:
            s = max(1.0, seq_len / self.original_max_position_embeddings)
            if seq_len == self._dynamic_seq_len_cached and s == self._dynamic_s_cached:
                return (
                    self._dynamic_cos_cached.to(dtype=x.dtype),
                    self._dynamic_sin_cached.to(dtype=x.dtype),
                )
            inv_freq = self._compute_inv_freq(s)
            t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            cos, sin = _interleave_cos_sin(freqs)
            cos = cos.to(dtype=x.dtype)
            sin = sin.to(dtype=x.dtype)
            self._dynamic_seq_len_cached = seq_len
            self._dynamic_s_cached = s
            self._dynamic_cos_cached = cos.detach()
            self._dynamic_sin_cached = sin.detach()
            return cos, sin
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=x.dtype),
                self.sin_cached[:seq_len].to(dtype=x.dtype),
            )


# ---------------------------------------------------------------------------- #
#  YaRN  (Yet Another RoPE extensioN)                                          #
# ---------------------------------------------------------------------------- #


class LlamaYarnScaledRotaryEmbedding(LlamaNTKByPartsScaledRotaryEmbedding):
    """YaRN (Yet Another RoPE extensioN) with attention temperature scaling.

    Combines NTK-by-parts frequency blending with attention temperature
    scaling (t = 1 + 0.1 * ln(s)) to improve extrapolation performance beyond
    the training distribution. The temperature scaling helps maintain attention
    distribution stability when processing sequences longer than seen during training.

    Attributes:
        attention_scaling (float): Temperature scaling factor (static mode).
            Computed as 1.0 + 0.1 * ln(scaling_factor).
        Inherits all attributes from LlamaNTKByPartsScaledRotaryEmbedding.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=2048,
        alpha=1,
        beta=32,
        dynamic=False,
    ):
        """Initialize YaRN scaled rotary position embedding.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Extension ratio for static mode. Defaults to 1.0.
            original_max_position_embeddings (int): Model's original context length.
                Defaults to 2048.
            alpha (float): Lower boundary for frequency blending. Defaults to 1.
            beta (float): Upper boundary for frequency blending. Defaults to 32.
            dynamic (bool): Whether to use dynamic scaling. Defaults to False.
        """
        if not dynamic:
            self.attention_scaling = 1.0 + 0.1 * math.log(scaling_factor)
        else:
            self.attention_scaling = None

        super().__init__(
            dim,
            max_position_embeddings,
            base,
            device,
            scaling_factor,
            original_max_position_embeddings,
            alpha,
            beta,
            dynamic,
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """Compute and cache cos/sin values with YaRN scaling including attention temperature.

        Applies NTK-by-parts frequency blending then scales results by the
        attention temperature factor to stabilize attention distributions.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = _interleave_cos_sin(freqs)
        if self.attention_scaling is not None:
            cos = cos * self.attention_scaling
            sin = sin * self.attention_scaling
        self.register_buffer("cos_cached", cos.contiguous().to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.contiguous().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """Forward pass to retrieve or compute YaRN scaled rotary embeddings.

        Combines NTK-by-parts blending with attention temperature scaling.
        In dynamic mode, computes temperature as 1.0 + 0.1 * ln(s).

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: YaRN-scaled cosine values of shape (seq_len, dim).
                - sin: YaRN-scaled sine values of shape (seq_len, dim).
        """
        if seq_len is None:
            seq_len = x.shape[2]
        if self.dynamic:
            s = max(1.0, seq_len / self.original_max_position_embeddings)
            if seq_len == self._dynamic_seq_len_cached and s == self._dynamic_s_cached:
                return (
                    self._dynamic_cos_cached.to(dtype=x.dtype),
                    self._dynamic_sin_cached.to(dtype=x.dtype),
                )
            attention_scaling = 1.0 + 0.1 * math.log(s)
            inv_freq = self._compute_inv_freq(s)
            t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            cos, sin = _interleave_cos_sin(freqs)
            cos = (cos * attention_scaling).to(dtype=x.dtype)
            sin = (sin * attention_scaling).to(dtype=x.dtype)
            self._dynamic_seq_len_cached = seq_len
            self._dynamic_s_cached = s
            self._dynamic_cos_cached = cos.detach()
            self._dynamic_sin_cached = sin.detach()
            return cos, sin
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=x.dtype),
                self.sin_cached[:seq_len].to(dtype=x.dtype),
            )


# ============================================================================ #
#  Shared helper for layer-aware attention temperature scaling                 #
#  Used by LlamaMyScaledRotaryEmbedding, LlamaMyScaledRotaryEmbedding2, and    #
#  LlamaBlockLayeredScaledRotaryEmbedding.                                     #
# ============================================================================ #


def _layer_aware_attn_scale(
    layer_idx: int,
    num_hidden_layers: int,
    seq_len: int,
    original_max_position_embeddings: int,
) -> float:
    """Compute layer-dependent attention amplitude scalar with inverted-U profile.

    Calculates an attention temperature scaling factor that varies across layers
    following an inverted-U (parabolic) profile. Middle layers receive weaker
    correction (approaching 1.0), while first and last layers receive stronger
    correction to compensate for their different roles in the transformer stack.

    The formula produces values in range [1.0, 1.1] where:
    - Layer 0 and last layer: maximum correction (~1.1)
    - Middle layers: minimum correction (~1.0)

    Args:
        layer_idx (int): Index of the current attention layer (0-based).
        num_hidden_layers (int): Total number of transformer layers.
        seq_len (int): Current sequence length being processed.
        original_max_position_embeddings (int): Model's original training
            context length L_0.

    Returns:
        float: Attention scaling factor suitable for multiplication with
            cos/sin tensors. Value is in [1.0, ~1.1].
    """
    layer_norm = 2.0 * layer_idx / max(num_hidden_layers - 1, 1) - 1.0
    u_norm = 1.0 - layer_norm**2
    layer_alpha = 0.1 * (1.0 - u_norm)
    return 1.0 + layer_alpha * math.log(
        max(1.0, seq_len / original_max_position_embeddings)
    )


# ============================================================================ #
#  My RoPE  (unified static / dynamic)                                         #
#                                                                              #
#  Each family follows the same two-class split as NTK-by-parts / YaRN:        #
#                                                                              #
#    Base class  — pure position encoding, no attention scaling.               #
#    Scaled class — inherits base, adds layer-aware attention temperature.     #
#                                                                              #
#    dynamic=False (default) — static mode: scaling_factor is fixed; cos/sin   #
#                              are pre-cached at init.                         #
#    dynamic=True            — dynamic mode: S = max(seq_len, L) / L is        #
#                              derived at runtime; no pre-caching.             #
# ============================================================================ #


class LlamaMyRotaryEmbedding(nn.Module):
    """Layer-aware My RoPE with position encoding only.

    Implements NTK-by-parts frequency blending with layer-adaptive alpha/beta
    boundaries using an inverted-U profile across layers. Different layers get
    different frequency blending parameters, allowing the model to adapt its
    position encoding strategy based on layer depth.

    Does NOT apply attention temperature correction (see LlamaMyScaledRotaryEmbedding
    for the version with attention scaling).

    Attributes:
        dim (int): Dimension of the embedding.
        max_position_embeddings (int): Maximum sequence length for caching.
        base (int): Base frequency for computing inverse frequencies.
        scaling_factor (float): Extension ratio for static mode.
        N (int): Total number of transformer layers.
        original_max_position_embeddings (int): Model's original context length.
        layer_idx (int): Index of this attention layer.
        alpha (float): Alpha parameter for layer adaptation (affects blending boundaries).
        dynamic (bool): Whether to use dynamic scaling.
        inv_freq (torch.Tensor): Inverse frequency buffer (static mode).
        inv_freq_base (torch.Tensor): Base inverse frequency (dynamic mode).
        w_ext (torch.Tensor): Blending weights (stored as _w_ext_layer).
        cos_cached (torch.Tensor): Cached cosine values (static mode only).
        sin_cached (torch.Tensor): Cached sine values (static mode only).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device=None,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 2048,
        layer_idx: int = 0,
        num_hidden_layers: int = 32,
        alpha: float = 0.2,
        dynamic: bool = False,
    ):
        """Initialize layer-aware My RoPE without attention scaling.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Extension ratio for static mode. Defaults to 1.0.
            original_max_position_embeddings (int): Model's original context length.
                Defaults to 2048.
            layer_idx (int): Index of this attention layer. Defaults to 0.
            num_hidden_layers (int): Total number of transformer layers.
                Defaults to 32.
            alpha (float): Layer adaptation parameter affecting blending boundaries.
                Defaults to 0.2.
            dynamic (bool): Whether to use dynamic scaling. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.N = num_hidden_layers
        self.original_max_position_embeddings = original_max_position_embeddings
        self.layer_idx = layer_idx
        self.alpha = alpha
        self.dynamic = dynamic

        theta_d = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )

        lambda_d = 2 * math.pi / theta_d
        r_d = original_max_position_embeddings / lambda_d

        layer_norm = 2.0 * layer_idx / (num_hidden_layers - 1) - 1.0
        u_norm = 1.0 - layer_norm**2
        alpha_eff = 1.0 + 1.0 * u_norm
        beta_eff = 32.0 + 8.0 * u_norm
        w_ext_layer = torch.clamp(
            (r_d - alpha_eff) / (beta_eff - alpha_eff),
            0.0,
            1.0,
        )
        self.register_buffer("_w_ext_layer", w_ext_layer, persistent=False)

        self._dynamic_seq_len_cached = -1
        self._dynamic_s_cached = -1.0
        self._dynamic_cos_cached = None
        self._dynamic_sin_cached = None

        if not dynamic:
            inv_freq = (
                w_ext_layer * theta_d + (1.0 - w_ext_layer) * theta_d / scaling_factor
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=self.inv_freq.device,
                dtype=torch.get_default_dtype(),
            )
        else:
            self.register_buffer("inv_freq_base", theta_d, persistent=False)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """Compute and cache cos/sin values with layer-adaptive blending.

        Uses layer-specific blending weights computed from inverted-U profile
        to adjust frequency interpolation per dimension.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = _interleave_cos_sin(freqs)
        self.register_buffer(
            "cos_cached",
            cos.contiguous().to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            sin.contiguous().to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass to retrieve or compute layer-aware My RoPE embeddings.

        Applies layer-adaptive NTK-by-parts blending. In dynamic mode,
        recomputes blending weights based on runtime scaling factor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Layer-adapted cosine values of shape (seq_len, dim).
                - sin: Layer-adapted sine values of shape (seq_len, dim).
        """
        if seq_len is None:
            seq_len = x.shape[2]
        if self.dynamic:
            device, dtype = x.device, x.dtype
            S = (
                max(seq_len, self.original_max_position_embeddings)
                / self.original_max_position_embeddings
            )
            if seq_len == self._dynamic_seq_len_cached and S == self._dynamic_s_cached:
                return (
                    self._dynamic_cos_cached.to(dtype=dtype),
                    self._dynamic_sin_cached.to(dtype=dtype),
                )
            inv_freq = (
                self._w_ext_layer.to(device=device) * self.inv_freq_base
                + (1.0 - self._w_ext_layer.to(device=device)) * self.inv_freq_base / S
            )
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            cos, sin = _interleave_cos_sin(freqs)
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
            self._dynamic_seq_len_cached = seq_len
            self._dynamic_s_cached = S
            self._dynamic_cos_cached = cos.detach()
            self._dynamic_sin_cached = sin.detach()
            return cos, sin
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=x.dtype),
                self.sin_cached[:seq_len].to(dtype=x.dtype),
            )


# ---------------------------------------------------------------------------- #


class LlamaMyScaledRotaryEmbedding(LlamaMyRotaryEmbedding):
    """Layer-aware My RoPE with attention temperature scaling.

    Inherits position encoding from LlamaMyRotaryEmbedding and applies a
    layer-dependent attention temperature scalar with inverted-U profile.
    The temperature scaling helps maintain attention stability for extended contexts.

    Attributes:
        Inherits all attributes from LlamaMyRotaryEmbedding.
    """

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """Compute and cache cos/sin values with layer-aware attention scaling.

        Extends parent class by multiplying cached embeddings with the
        layer-dependent attention temperature factor.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = _interleave_cos_sin(freqs)
        attn_scale = _layer_aware_attn_scale(
            self.layer_idx,
            self.N,
            seq_len,
            self.original_max_position_embeddings,
        )
        self.register_buffer(
            "cos_cached",
            (cos * attn_scale).contiguous().to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (sin * attn_scale).contiguous().to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass with layer-aware attention temperature scaling.

        In dynamic mode, computes position embeddings first, then applies
        layer-dependent temperature scaling factor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Scaled cosine values of shape (seq_len, dim).
                - sin: Scaled sine values of shape (seq_len, dim).
        """
        if self.dynamic:
            if seq_len is None:
                seq_len = x.shape[2]
            cos, sin = super().forward(x, seq_len)
            attn_scale = _layer_aware_attn_scale(
                self.layer_idx,
                self.N,
                seq_len,
                self.original_max_position_embeddings,
            )
            return (cos * attn_scale).to(x.dtype), (sin * attn_scale).to(x.dtype)
        return super().forward(x, seq_len)


# ---------------------------------------------------------------------------- #


class LlamaMyRotaryEmbedding2(nn.Module):
    """Multi-scale My RoPE with position encoding only.

    Splits the head dimension into multiple sub-spaces (local, paragraph, document),
    each operating at different temporal scales with independent NTK-by-parts
    parameters and base frequencies. This allows the model to capture both
    fine-grained local patterns and coarse-grained global structure simultaneously.

    Does NOT apply attention temperature correction (see LlamaMyScaledRotaryEmbedding2
    for the version with attention scaling).

    Attributes:
        dim (int): Total dimension of the embedding.
        base (int): Base frequency for computing inverse frequencies.
        max_position_embeddings (int): Maximum sequence length for caching.
        original_max_position_embeddings (int): Model's original context length.
        scaling_factor (float): Extension ratio for static mode.
        alpha (float): Alpha parameter (reserved for future use).
        dynamic (bool): Whether to use dynamic scaling.
        layer_idx (int): Index of this attention layer.
        N (int): Total number of transformer layers.
        scales (list[dict]): Configuration for each sub-space scale defining
            base frequency, window size, and dimension ratio.
        scale_buffers (list[dict]): Buffers for each sub-space containing
            inverse frequencies and dimension ranges.
        cos_cached (torch.Tensor): Cached cosine values (static mode only).
        sin_cached (torch.Tensor): Cached sine values (static mode only).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device=None,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 2048,
        layer_idx: int = 0,
        num_hidden_layers: int = 32,
        alpha: float = 0.2,
        dynamic: bool = False,
    ):
        """Initialize multi-scale My RoPE 2 without attention scaling.

        Args:
            dim (int): Total dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Extension ratio for static mode. Defaults to 1.0.
            original_max_position_embeddings (int): Model's original context length.
                Defaults to 2048.
            layer_idx (int): Index of this attention layer. Defaults to 0.
            num_hidden_layers (int): Total number of transformer layers.
                Defaults to 32.
            alpha (float): Reserved parameter for future extensions. Defaults to 0.2.
            dynamic (bool): Whether to use dynamic scaling. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scaling_factor = scaling_factor
        self.alpha = alpha
        self.dynamic = dynamic
        self.layer_idx = layer_idx
        self.N = num_hidden_layers

        self.scales = [
            {"base": base, "window": 512, "dim_ratio": 0.4},
            {"base": base * 4, "window": 2048, "dim_ratio": 0.3},
            {"base": base * 16, "window": 8192, "dim_ratio": 0.3},
        ]
        self.scale_buffers = []
        current_dim = 0
        for i, scale in enumerate(self.scales):
            scale_dim = int(dim * scale["dim_ratio"])
            scale_dim = scale_dim - (scale_dim % 2)
            if i == len(self.scales) - 1:
                scale_dim = dim - current_dim
            inv_freq = 1.0 / (
                scale["base"]
                ** (torch.arange(0, scale_dim, 2, device=device).float() / scale_dim)
            )
            self.register_buffer(f"inv_freq_scale_{i}", inv_freq, persistent=False)
            self.scale_buffers.append(
                {
                    "inv_freq": inv_freq,
                    "window": scale["window"],
                    "dim_start": current_dim,
                    "dim_end": current_dim + scale_dim,
                }
            )
            current_dim += scale_dim

        self._ntk_params = [
            (0.8, 24.0),
            (1.0, 32.0),
            (1.2, 40.0),
        ]

        for i, (buffer, (ntk_a, ntk_b)) in enumerate(
            zip(self.scale_buffers, self._ntk_params)
        ):
            theta_d = buffer["inv_freq"]
            window = buffer["window"]
            lambda_d = 2 * math.pi / theta_d
            r_d = window / lambda_d
            w_ext = torch.clamp((r_d - ntk_a) / (ntk_b - ntk_a), 0.0, 1.0)
            self.register_buffer(f"_w_ext_{i}", w_ext, persistent=False)

        self._dynamic_seq_len_cached = -1
        self._dynamic_cos_cached = None
        self._dynamic_sin_cached = None

        if not dynamic:
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=device or self.scale_buffers[0]["inv_freq"].device,
                dtype=torch.get_default_dtype(),
            )

    def _get_scale_inv_freq_static(self, scale_idx: int, device):
        """Compute inverse frequencies for a specific sub-space in static mode.

        Applies NTK-by-parts blending using the static scaling factor
        for the specified sub-space index.

        Args:
            scale_idx (int): Index of the sub-space (0, 1, or 2).
            device (torch.device): Device to place result on.

        Returns:
            torch.Tensor: Blended inverse frequencies for the sub-space.
        """
        buffer = self.scale_buffers[scale_idx]
        theta_d = buffer["inv_freq"].to(device=device)
        w_ext = getattr(self, f"_w_ext_{scale_idx}").to(device=device)
        return w_ext * theta_d + (1.0 - w_ext) * theta_d / self.scaling_factor

    def _get_scale_inv_freq_dynamic(self, scale_idx: int, S: float, device):
        """Compute inverse frequencies for a specific sub-space in dynamic mode.

        Applies NTK-by-parts blending using the runtime scaling factor S
        for the specified sub-space index.

        Args:
            scale_idx (int): Index of the sub-space (0, 1, or 2).
            S (float): Runtime scaling factor.
            device (torch.device): Device to place result on.

        Returns:
            torch.Tensor: Blended inverse frequencies for the sub-space.
        """
        buffer = self.scale_buffers[scale_idx]
        theta_d = buffer["inv_freq"]
        window = buffer["window"]
        w_ext = getattr(self, f"_w_ext_{scale_idx}")
        return (w_ext * theta_d + (1.0 - w_ext) * theta_d / S).to(device=device)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache multi-scale cos/sin values.

        Builds the full embedding by concatenating contributions from each
        sub-space, each processed with its own frequency parameters.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        cos_cached = torch.zeros(seq_len, self.dim, device=device)
        sin_cached = torch.zeros(seq_len, self.dim, device=device)
        for i, buffer in enumerate(self.scale_buffers):
            scale_dim = buffer["dim_end"] - buffer["dim_start"]
            if scale_dim <= 0:
                continue
            inv_freq_scaled = self._get_scale_inv_freq_static(i, device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq_scaled.dtype)
            freqs = torch.outer(t, inv_freq_scaled)
            emb = torch.cat((freqs, freqs), dim=-1)
            ds = buffer["dim_start"]
            de = buffer["dim_end"]
            cos_cached[:, ds:de] = emb.cos()
            sin_cached[:, ds:de] = emb.sin()

        self.register_buffer(
            "cos_cached",
            cos_cached.contiguous().to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            sin_cached.contiguous().to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass to retrieve or compute multi-scale My RoPE 2 embeddings.

        Processes each sub-space independently with its own scaling parameters,
        then concatenates results into the full embedding dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Multi-scale cosine values of shape (seq_len, dim).
                - sin: Multi-scale sine values of shape (seq_len, dim).
        """
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            if seq_len == self._dynamic_seq_len_cached:
                return (
                    self._dynamic_cos_cached.to(dtype=dtype),
                    self._dynamic_sin_cached.to(dtype=dtype),
                )
            cos_cached = torch.zeros(seq_len, self.dim, device=device)
            sin_cached = torch.zeros(seq_len, self.dim, device=device)
            for i, buffer in enumerate(self.scale_buffers):
                scale_dim = buffer["dim_end"] - buffer["dim_start"]
                if scale_dim <= 0:
                    continue
                s = max(1.0, seq_len / buffer["window"])
                inv_freq_scaled = self._get_scale_inv_freq_dynamic(i, s, device)
                t = torch.arange(seq_len, device=device, dtype=inv_freq_scaled.dtype)
                freqs = torch.outer(t, inv_freq_scaled)
                emb = torch.cat((freqs, freqs), dim=-1)
                ds = buffer["dim_start"]
                de = buffer["dim_end"]
                cos_cached[:, ds:de] = emb.cos()
                sin_cached[:, ds:de] = emb.sin()

            cos_cached = cos_cached.to(dtype=dtype)
            sin_cached = sin_cached.to(dtype=dtype)
            self._dynamic_seq_len_cached = seq_len
            self._dynamic_cos_cached = cos_cached.detach()
            self._dynamic_sin_cached = sin_cached.detach()
            return cos_cached, sin_cached
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=dtype),
                self.sin_cached[:seq_len].to(dtype=dtype),
            )


# ---------------------------------------------------------------------------- #


class LlamaMyScaledRotaryEmbedding2(LlamaMyRotaryEmbedding2):
    """Multi-scale My RoPE 2 with attention temperature scaling.

    Inherits position encoding from LlamaMyRotaryEmbedding2 and applies the
    same layer-dependent attention temperature scalar with inverted-U profile.

    Attributes:
        Inherits all attributes from LlamaMyRotaryEmbedding2.
    """

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache multi-scale cos/sin values with attention scaling.

        Extends parent class by applying layer-dependent temperature factor
        to the concatenated multi-scale embeddings.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        cos_cached = torch.zeros(seq_len, self.dim, device=device)
        sin_cached = torch.zeros(seq_len, self.dim, device=device)
        for i, buffer in enumerate(self.scale_buffers):
            scale_dim = buffer["dim_end"] - buffer["dim_start"]
            if scale_dim <= 0:
                continue
            inv_freq_scaled = self._get_scale_inv_freq_static(i, device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq_scaled.dtype)
            freqs = torch.outer(t, inv_freq_scaled)
            emb = torch.cat((freqs, freqs), dim=-1)
            ds = buffer["dim_start"]
            de = buffer["dim_end"]
            cos_cached[:, ds:de] = emb.cos()
            sin_cached[:, ds:de] = emb.sin()

        attn_scale = _layer_aware_attn_scale(
            self.layer_idx,
            self.N,
            seq_len,
            self.original_max_position_embeddings,
        )

        self.register_buffer(
            "cos_cached",
            (cos_cached * attn_scale).contiguous().to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (sin_cached * attn_scale).contiguous().to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass with multi-scale position encoding and attention scaling.

        Computes multi-scale embeddings first, then applies layer-dependent
        temperature scaling factor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Scaled multi-scale cosine values of shape (seq_len, dim).
                - sin: Scaled multi-scale sine values of shape (seq_len, dim).
        """
        if self.dynamic:
            if seq_len is None:
                seq_len = x.shape[2]
            cos, sin = super().forward(x, seq_len)
            attn_scale = _layer_aware_attn_scale(
                self.layer_idx,
                self.N,
                seq_len,
                self.original_max_position_embeddings,
            )
            return (cos * attn_scale).to(x.dtype), (sin * attn_scale).to(x.dtype)
        return super().forward(x, seq_len)


# ============================================================================ #
#  Block-Layered RoPE                                                          #
# ============================================================================ #


class LlamaBlockLayeredRotaryEmbedding(nn.Module):
    """Block-Layered RoPE with position encoding only.

    Implements quantized effective position indices using per-dimension block
    sizes that grow exponentially with dimension index. This prevents angular
    value out-of-distribution issues when extending beyond original context length.

    The key insight is that different frequency dimensions should quantize
    positions at different rates: high frequencies need finer granularity
    (smaller blocks), while low frequencies can use coarser quantization
    (larger blocks) without losing information.

    Does NOT apply attention temperature correction (see LlamaBlockLayeredScaledRotaryEmbedding
    for the version with attention scaling).

    Attributes:
        dim (int): Dimension of the embedding.
        base (int): Base frequency for computing inverse frequencies.
        max_position_embeddings (int): Maximum sequence length for caching.
        original_max_position_embeddings (int): Model's original context length.
        scaling_factor (float): Extension ratio for static mode.
        layer_idx (int): Index of this attention layer.
        N (int): Total number of transformer layers.
        dynamic (bool): Whether to use dynamic scaling.
        i_star (int): Critical dimension index separating high/low frequency behavior.
        inv_freq (torch.Tensor): Inverse frequency buffer.
        block_sizes (torch.Tensor): Per-dimension block sizes (static mode).
        cos_cached (torch.Tensor): Cached cosine values (static mode only).
        sin_cached (torch.Tensor): Cached sine values (static mode only).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device=None,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 2048,
        layer_idx: int = 0,
        num_hidden_layers: int = 32,
        dynamic: bool = False,
    ):
        """Initialize Block-Layered RoPE without attention scaling.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Extension ratio for static mode. Defaults to 1.0.
            original_max_position_embeddings (int): Model's original context length.
                Defaults to 2048.
            layer_idx (int): Index of this attention layer. Defaults to 0.
            num_hidden_layers (int): Total number of transformer layers.
                Defaults to 32.
            dynamic (bool): Whether to use dynamic scaling. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scaling_factor = scaling_factor
        self.layer_idx = layer_idx
        self.N = num_hidden_layers
        self.dynamic = dynamic

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        r_d = original_max_position_embeddings * inv_freq / (2.0 * math.pi)
        i_star = int((r_d >= 1.0).sum().item())
        self.i_star = max(1, min(i_star, dim // 2 - 1))

        self._dynamic_seq_len_cached = -1
        self._dynamic_s_cached = -1.0
        self._dynamic_cos_cached = None
        self._dynamic_sin_cached = None

        if not dynamic:
            block_sizes = self._compute_block_sizes(scaling_factor, device=device)
            self.register_buffer("block_sizes", block_sizes, persistent=False)
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=device or inv_freq.device,
                dtype=torch.get_default_dtype(),
            )

    def _compute_block_sizes(self, S: float, device=None) -> torch.Tensor:
        """Compute exponential block size schedule for all dimensions.

        Generates per-dimension block sizes following b_i = S^(i/i_star) for i < i_star,
        and b_i = S for i >= i_star. This creates smoothly accelerating block growth.

        Args:
            S (float): Scaling factor determining maximum block size.
            device (torch.device, optional): Device to place result on. Defaults to None.

        Returns:
            torch.Tensor: Block sizes of shape (dim//2,), values in [1.0, S].
        """
        half_dim = self.dim // 2
        indices = torch.arange(half_dim, device=device, dtype=torch.float32)
        exponent = torch.clamp(indices / float(self.i_star), 0.0, 1.0)
        b = torch.clamp(torch.pow(float(S), exponent), min=1.0, max=float(S))
        return b

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache cos/sin values with block-layered quantization.

        Applies per-dimension block quantization before computing frequencies,
        effectively compressing the position space differently for each dimension.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        b = self.block_sizes.to(device=device)
        inv_freq = self.inv_freq.to(device=device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        t_eff = torch.floor_divide(t.unsqueeze(1), b.unsqueeze(0))
        freqs = t_eff * inv_freq.unsqueeze(0)
        cos, sin = _interleave_cos_sin(freqs)
        self.register_buffer("cos_cached", cos.contiguous().to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.contiguous().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass to retrieve or compute block-layered RoPE embeddings.

        Applies exponential block-size quantization to create dimension-dependent
        effective position indices before computing rotary embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Block-quantized cosine values of shape (seq_len, dim).
                - sin: Block-quantized sine values of shape (seq_len, dim).
        """
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            S = max(1.0, seq_len / self.original_max_position_embeddings)
            if seq_len == self._dynamic_seq_len_cached and S == self._dynamic_s_cached:
                return (
                    self._dynamic_cos_cached.to(dtype=dtype),
                    self._dynamic_sin_cached.to(dtype=dtype),
                )
            b = self._compute_block_sizes(S, device=device)
            inv_freq = self.inv_freq.to(device=device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            t_eff = torch.floor_divide(t.unsqueeze(1), b.unsqueeze(0))
            freqs = t_eff * inv_freq.unsqueeze(0)
            cos, sin = _interleave_cos_sin(freqs)
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
            self._dynamic_seq_len_cached = seq_len
            self._dynamic_s_cached = S
            self._dynamic_cos_cached = cos.detach()
            self._dynamic_sin_cached = sin.detach()
            return cos, sin
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=dtype),
                self.sin_cached[:seq_len].to(dtype=dtype),
            )


# ---------------------------------------------------------------------------- #


class LlamaBlockLayeredScaledRotaryEmbedding(LlamaBlockLayeredRotaryEmbedding):
    """Block-Layered RoPE with attention temperature scaling.

    Inherits position encoding from LlamaBlockLayeredRotaryEmbedding and
    applies a layer-dependent attention temperature scalar with inverted-U profile.

    Attributes:
        Inherits all attributes from LlamaBlockLayeredRotaryEmbedding.
    """

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache cos/sin values with block quantization and attention scaling.

        Extends parent class by applying layer-dependent temperature factor
        to the block-quantized embeddings.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        b = self.block_sizes.to(device=device)
        inv_freq = self.inv_freq.to(device=device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        t_eff = torch.floor_divide(t.unsqueeze(1), b.unsqueeze(0))
        freqs = t_eff * inv_freq.unsqueeze(0)
        cos, sin = _interleave_cos_sin(freqs)
        attn_scale = _layer_aware_attn_scale(
            self.layer_idx,
            self.N,
            seq_len,
            self.original_max_position_embeddings,
        )
        self.register_buffer(
            "cos_cached",
            (cos * attn_scale).contiguous().to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (sin * attn_scale).contiguous().to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass with block-layered quantization and attention scaling.

        Computes block-quantized embeddings first, then applies layer-dependent
        temperature scaling factor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Scaled block-quantized cosine values of shape (seq_len, dim).
                - sin: Scaled block-quantized sine values of shape (seq_len, dim).
        """
        if self.dynamic:
            if seq_len is None:
                seq_len = x.shape[2]
            cos, sin = super().forward(x, seq_len)
            attn_scale = _layer_aware_attn_scale(
                self.layer_idx,
                self.N,
                seq_len,
                self.original_max_position_embeddings,
            )
            return (cos * attn_scale).to(x.dtype), (sin * attn_scale).to(x.dtype)
        return super().forward(x, seq_len)


# ============================================================================ #
#  Freq-Smooth RoPE                                                            #
# ============================================================================ #


class LlamaFreqSmoothRotaryEmbedding(nn.Module):
    """Freq-Smooth Block RoPE with position encoding only.

    Implements quantized effective position indices using a quadratic block-size
    schedule derived from normalized RoPE base frequencies. Provides C1 smoothness
    (continuous first derivative) at the critical dimension boundary i_star,
    avoiding discontinuities in the block-size function.

    Unlike Block-Layered which uses exponential growth, Freq-Smooth uses a
    parabolic schedule tied to the actual frequency values, providing smoother
    transitions between quantization regimes.

    Does NOT apply attention temperature correction (see LlamaFreqSmoothScaledRotaryEmbedding
    for the version with attention scaling).

    Attributes:
        dim (int): Dimension of the embedding.
        base (int): Base frequency for computing inverse frequencies.
        max_position_embeddings (int): Maximum sequence length for caching.
        original_max_position_embeddings (int): Model's original context length.
        scaling_factor (float): Extension ratio for static mode.
        layer_idx (int): Index of this attention layer.
        N (int): Total number of transformer layers.
        dynamic (bool): Whether to use dynamic scaling.
        i_star (int): Critical dimension index separating high/low frequency behavior.
        theta_istar (float): Frequency value at critical dimension i_star.
        inv_freq (torch.Tensor): Inverse frequency buffer.
        block_sizes (torch.Tensor): Per-dimension block sizes (static mode).
        cos_cached (torch.Tensor): Cached cosine values (static mode only).
        sin_cached (torch.Tensor): Cached sine values (static mode only).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device=None,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 2048,
        layer_idx: int = 0,
        num_hidden_layers: int = 32,
        dynamic: bool = False,
    ):
        """Initialize Freq-Smooth RoPE without attention scaling.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Extension ratio for static mode. Defaults to 1.0.
            original_max_position_embeddings (int): Model's original context length.
                Defaults to 2048.
            layer_idx (int): Index of this attention layer. Defaults to 0.
            num_hidden_layers (int): Total number of transformer layers.
                Defaults to 32.
            dynamic (bool): Whether to use dynamic scaling. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scaling_factor = scaling_factor
        self.layer_idx = layer_idx
        self.N = num_hidden_layers
        self.dynamic = dynamic

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        r = original_max_position_embeddings * inv_freq / (2.0 * math.pi)
        i_star = int((r >= 1.0).sum().item())
        self.i_star = max(1, min(i_star, dim // 2 - 1))
        self.theta_istar: float = float(inv_freq[self.i_star].item())

        self._denom = max(1.0 - self.theta_istar, 1e-8)

        self._dynamic_seq_len_cached = -1
        self._dynamic_s_cached = -1.0
        self._dynamic_cos_cached = None
        self._dynamic_sin_cached = None

        if not dynamic:
            block_sizes = self._compute_block_sizes(scaling_factor, device=device)
            self.register_buffer("block_sizes", block_sizes, persistent=False)
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=device or inv_freq.device,
                dtype=torch.get_default_dtype(),
            )

    def _compute_block_sizes(self, S: float, device=None) -> torch.Tensor:
        """Compute quadratic (parabolic) block size schedule with C1 smoothness.

        Generates per-dimension block sizes using normalized frequencies to create
        a smooth parabolic transition: b_i = S - (S-1) * hat_theta_i^2 for i < i_star,
        and b_i = S for i >= i_star. This ensures continuous derivative at i_star.

        Args:
            S (float): Scaling factor determining maximum block size.
            device (torch.device, optional): Device to place result on. Defaults to None.

        Returns:
            torch.Tensor: Smooth block sizes of shape (dim//2,), values in [1.0, S].
        """
        theta = self.inv_freq.to(device=device)
        S_val = float(S)
        th_star = self.theta_istar
        denom = self._denom
        theta_hat = torch.clamp(
            (theta - th_star) / denom,
            min=0.0,
            max=1.0,
        )
        b = S_val - (S_val - 1.0) * theta_hat * theta_hat
        b[self.i_star :] = S_val
        return torch.clamp(b, min=1.0, max=S_val)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache cos/sin values with smooth block quantization.

        Applies parabolic block-size schedule before computing frequencies,
        ensuring C1 continuity at the critical dimension boundary.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        b = self.block_sizes.to(device=device)
        inv_freq = self.inv_freq.to(device=device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        t_eff = torch.floor_divide(t.unsqueeze(1), b.unsqueeze(0))
        freqs = t_eff * inv_freq.unsqueeze(0)
        cos, sin = _interleave_cos_sin(freqs)
        self.register_buffer("cos_cached", cos.contiguous().to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.contiguous().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass to retrieve or compute Freq-Smooth RoPE embeddings.

        Applies parabolic block-size quantization with C1 smoothness for
        stable gradient flow near the critical dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Smooth block-quantized cosine values of shape (seq_len, dim).
                - sin: Smooth block-quantized sine values of shape (seq_len, dim).
        """
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            S = max(1.0, seq_len / self.original_max_position_embeddings)
            if seq_len == self._dynamic_seq_len_cached and S == self._dynamic_s_cached:
                return (
                    self._dynamic_cos_cached.to(dtype=dtype),
                    self._dynamic_sin_cached.to(dtype=dtype),
                )
            b = self._compute_block_sizes(S, device=device)
            inv_freq = self.inv_freq.to(device=device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            t_eff = torch.floor_divide(t.unsqueeze(1), b.unsqueeze(0))
            freqs = t_eff * inv_freq.unsqueeze(0)
            cos, sin = _interleave_cos_sin(freqs)
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
            self._dynamic_seq_len_cached = seq_len
            self._dynamic_s_cached = S
            self._dynamic_cos_cached = cos.detach()
            self._dynamic_sin_cached = sin.detach()
            return cos, sin
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=dtype),
                self.sin_cached[:seq_len].to(dtype=dtype),
            )


# ---------------------------------------------------------------------------- #


class LlamaFreqSmoothScaledRotaryEmbedding(LlamaFreqSmoothRotaryEmbedding):
    """Freq-Smooth Block RoPE with attention temperature scaling.

    Inherits position encoding from LlamaFreqSmoothRotaryEmbedding and applies
    a layer-dependent attention temperature scalar with inverted-U profile.

    Attributes:
        Inherits all attributes from LlamaFreqSmoothRotaryEmbedding.
    """

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache cos/sin values with smooth block quantization and attention scaling.

        Extends parent class by applying layer-dependent temperature factor
        to the smoothly quantized embeddings.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        b = self.block_sizes.to(device=device)
        inv_freq = self.inv_freq.to(device=device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        t_eff = torch.floor_divide(t.unsqueeze(1), b.unsqueeze(0))
        freqs = t_eff * inv_freq.unsqueeze(0)
        cos, sin = _interleave_cos_sin(freqs)
        attn_scale = _layer_aware_attn_scale(
            self.layer_idx,
            self.N,
            seq_len,
            self.original_max_position_embeddings,
        )
        self.register_buffer(
            "cos_cached",
            (cos * attn_scale).contiguous().to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (sin * attn_scale).contiguous().to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass with smooth block quantization and attention scaling.

        Computes parabolic-block-quantized embeddings first, then applies
        layer-dependent temperature scaling factor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Scaled smooth cosine values of shape (seq_len, dim).
                - sin: Scaled smooth sine values of shape (seq_len, dim).
        """
        if self.dynamic:
            if seq_len is None:
                seq_len = x.shape[2]
            cos, sin = super().forward(x, seq_len)
            attn_scale = _layer_aware_attn_scale(
                self.layer_idx,
                self.N,
                seq_len,
                self.original_max_position_embeddings,
            )
            return (cos * attn_scale).to(x.dtype), (sin * attn_scale).to(x.dtype)
        return super().forward(x, seq_len)


# ============================================================================ #
#  Freq-Reciprocal RoPE                                                        #
# ============================================================================ #


class LlamaFreqReciprocalRotaryEmbedding(nn.Module):
    """Freq-Reciprocal Block RoPE — position encoding only (no attention scaling).

    Core Idea
    ---------
    Like Block-Layered RoPE, each RoPE dimension i uses a quantised effective
    position index::

        t_eff(i) = floor(t / b_i)

    The block-size schedule b_i is defined so that its rate of change db_i/di
    and the rate of decay |dθ_i/di| of the RoPE base frequency satisfy a
    **constant-product (reciprocal)** relationship for all i < i*::

        db_i/di · |dθ_i/di| = const

    Because |dθ_i/di| ∝ θ_i (decelerating), the reciprocal constraint forces
    db_i/di ∝ 1/θ_i (accelerating) — matching the BlockLayered design
    philosophy while being grounded in the frequency structure of the model.

    Formula
    -------
    Let i* be the critical dimension (first index where r_i = L_0·θ_i/(2π) < 1).
    Let K = (S−1) / (1/θ_{i*} − 1). Then::

        b_i = 1 + K · (1/θ_i − 1)    for i < i*
        b_i = S                        for i ≥ i*

    Equivalently, using θ_i = base^{−2i/d}::

        b_i = 1 + (S−1) · (base^{2i/d} − 1) / (base^{2i*/d} − 1)

    Properties
    ----------
    * b_0 = 1  (i=0: 1/θ_0 = 1, so b_0 = 1 + K·0 = 1)
    * b_{i*} = S  (by definition of K)
    * b_i ∈ [1, S] for all i  (b is monotone increasing in i)
    * db_i/di = K · (2·ln·base/d) / θ_i  — accelerating, mirrors 1/θ_i growth
    * db_i/di · |dθ_i/di| = K · (2·ln·base/d)² = const  ✓
    * b_i = S exactly for all i ≥ i*  (hard-set, no floating-point residual)
    * No additional hyper-parameters beyond S
    * Degrades to standard RoPE when S = 1

    Relationship to BlockLayered
    ----------------------------
    BlockLayered uses b_i = S^{i/i*}, whose derivative grows as b_i · ln(S).
    Freq-Reciprocal uses db/di ∝ 1/θ_i = base^{2i/d}, which is also
    accelerating but tied to the model's own frequency grid rather than to S.
    This means the acceleration profile is independent of the choice of S,
    making the method more principled when S varies (e.g. in dynamic mode).

    Parameters
    ----------
    scaling_factor : float
        Static extension ratio S > 1.0. Ignored in dynamic mode.
    dynamic : bool
        False (default) — static mode: b_i and cos/sin pre-cached at init.
        True            — dynamic mode: S = max(1, seq_len / L_0) at runtime.
    layer_idx : int
        0-based index of this attention layer. Stored for use by
        ``LlamaFreqReciprocalScaledRotaryEmbedding``; not used here.
    num_hidden_layers : int
        Total transformer layers. Stored for the scaled subclass.

    Attributes:
        dim (int): Dimension of the embedding.
        base (int): Base frequency for computing inverse frequencies.
        max_position_embeddings (int): Maximum sequence length for caching.
        original_max_position_embeddings (int): Model's original context length.
        scaling_factor (float): Extension ratio for static mode.
        layer_idx (int): Index of this attention layer.
        N (int): Total number of transformer layers.
        dynamic (bool): Whether to use dynamic scaling.
        i_star (int): Critical dimension index.
        inv_theta_istar (float): Inverse of frequency at critical dimension.
        inv_freq (torch.Tensor): Inverse frequency buffer.
        block_sizes (torch.Tensor): Per-dimension block sizes (static mode).
        cos_cached (torch.Tensor): Cached cosine values (static mode only).
        sin_cached (torch.Tensor): Cached sine values (static mode only).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device=None,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 2048,
        layer_idx: int = 0,
        num_hidden_layers: int = 32,
        dynamic: bool = False,
    ):
        """Initialize Freq-Reciprocal RoPE without attention scaling.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Extension ratio for static mode. Defaults to 1.0.
            original_max_position_embeddings (int): Model's original context length.
                Defaults to 2048.
            layer_idx (int): Index of this attention layer. Defaults to 0.
            num_hidden_layers (int): Total number of transformer layers.
                Defaults to 32.
            dynamic (bool): Whether to use dynamic scaling. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scaling_factor = scaling_factor
        self.layer_idx = layer_idx
        self.N = num_hidden_layers
        self.dynamic = dynamic

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        r = original_max_position_embeddings * inv_freq / (2.0 * math.pi)
        i_star = int((r >= 1.0).sum().item())
        self.i_star = max(1, min(i_star, dim // 2 - 1))

        self.inv_theta_istar: float = float((base ** (2.0 * self.i_star / dim)))
        self._inv_theta_denom = self.inv_theta_istar - 1.0

        self._dynamic_seq_len_cached = -1
        self._dynamic_s_cached = -1.0
        self._dynamic_cos_cached = None
        self._dynamic_sin_cached = None

        if not dynamic:
            block_sizes = self._compute_block_sizes(scaling_factor, device=device)
            self.register_buffer("block_sizes", block_sizes, persistent=False)
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=device or inv_freq.device,
                dtype=torch.get_default_dtype(),
            )

    def _compute_block_sizes(self, S: float, device=None) -> torch.Tensor:
        """Compute reciprocal block size schedule for all dimensions.

        Generates per-dimension block sizes using the constant-product constraint:
        b_i = 1 + K * (1/theta_i - 1) for i < i*, where K = (S-1)/(1/theta_i* - 1).

        Args:
            S (float): Scaling factor determining maximum block size.
            device (torch.device, optional): Device to place result on. Defaults to None.

        Returns:
            torch.Tensor: Reciprocal block sizes of shape (dim//2,), values in [1.0, S].
        """
        S_val = float(S)
        inv_theta = 1.0 / self.inv_freq.to(device=device)
        denom = self._inv_theta_denom
        if abs(denom) < 1e-8:
            return torch.ones(self.dim // 2, device=device, dtype=torch.float32) * S_val
        K = (S_val - 1.0) / denom
        b = 1.0 + K * (inv_theta - 1.0)
        b[self.i_star :] = S_val
        return torch.clamp(b, min=1.0, max=S_val)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache cos/sin values with reciprocal block quantization.

        Applies reciprocal block-size schedule before computing frequencies,
        implementing the constant-product constraint between block growth and
        frequency decay.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        b = self.block_sizes.to(device=device)
        inv_freq = self.inv_freq.to(device=device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        t_eff = torch.floor_divide(t.unsqueeze(1), b.unsqueeze(0))
        freqs = t_eff * inv_freq.unsqueeze(0)
        cos, sin = _interleave_cos_sin(freqs)
        self.register_buffer("cos_cached", cos.contiguous().to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.contiguous().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass to retrieve or compute Freq-Reciprocal RoPE embeddings.

        Applies reciprocal block-size quantization that maintains constant-product
        relationship between block growth rate and frequency decay rate.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Reciprocal block-quantized cosine values of shape (seq_len, dim).
                - sin: Reciprocal block-quantized sine values of shape (seq_len, dim).
        """
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            S = max(1.0, seq_len / self.original_max_position_embeddings)
            if seq_len == self._dynamic_seq_len_cached and S == self._dynamic_s_cached:
                return (
                    self._dynamic_cos_cached.to(dtype=dtype),
                    self._dynamic_sin_cached.to(dtype=dtype),
                )
            b = self._compute_block_sizes(S, device=device)
            inv_freq = self.inv_freq.to(device=device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            t_eff = torch.floor_divide(t.unsqueeze(1), b.unsqueeze(0))
            freqs = t_eff * inv_freq.unsqueeze(0)
            cos, sin = _interleave_cos_sin(freqs)
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
            self._dynamic_seq_len_cached = seq_len
            self._dynamic_s_cached = S
            self._dynamic_cos_cached = cos.detach()
            self._dynamic_sin_cached = sin.detach()
            return cos, sin
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=dtype),
                self.sin_cached[:seq_len].to(dtype=dtype),
            )


# ---------------------------------------------------------------------------- #


class LlamaFreqReciprocalScaledRotaryEmbedding(LlamaFreqReciprocalRotaryEmbedding):
    """Freq-Reciprocal Block RoPE with power-law attention temperature scaling.

    Inherits position encoding from LlamaFreqReciprocalRotaryEmbedding and
    applies a power-law attention temperature scaling with layer-dependent factors.
    The scaling accounts for both position index and layer depth to provide
    fine-grained control over attention distribution stability.

    Attributes:
        alpha (float): Exponent for scaling factor S (currently unused, reserved).
        beta (float): Layer-dependent scaling coefficient (currently unused, reserved).
        Inherits all attributes from LlamaFreqReciprocalRotaryEmbedding.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device=None,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 2048,
        layer_idx: int = 0,
        num_hidden_layers: int = 32,
        dynamic: bool = False,
        alpha: float = 0.25,
        beta: float = 0.05,
    ):
        """Initialize Freq-Reciprocal RoPE with power-law attention scaling.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Extension ratio for static mode. Defaults to 1.0.
            original_max_position_embeddings (int): Model's original context length.
                Defaults to 2048.
            layer_idx (int): Index of this attention layer. Defaults to 0.
            num_hidden_layers (int): Total number of transformer layers.
                Defaults to 32.
            dynamic (bool): Whether to use dynamic scaling. Defaults to False.
            alpha (float): Reserved exponent parameter. Defaults to 0.25.
            beta (float): Reserved coefficient parameter. Defaults to 0.05.
        """
        super().__init__(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            device=device,
            scaling_factor=scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            layer_idx=layer_idx,
            num_hidden_layers=num_hidden_layers,
            dynamic=dynamic,
        )
        self.alpha = alpha
        self.beta = beta

    def _compute_attn_scale(self, seq_len: int, device):
        """Compute power-law attention scaling factor with layer dependence.

        Calculates position-dependent and layer-dependent scaling using logarithmic
        growth model: S_t = 1.0 + 0.1 * log(t/L_0) * (1 + depth_factor), where
        depth_factor varies exponentially with layer index.

        Args:
            seq_len (int): Current sequence length.
            device (torch.device): Device to place result on.

        Returns:
            torch.Tensor: Attention scaling factors of shape (seq_len, 1).
        """
        t = torch.maximum(
            torch.tensor(1.0, device=device),
            torch.arange(seq_len, device=device, dtype=torch.float32)
            / self.original_max_position_embeddings,
        )
        depth_factor = math.exp(self.layer_idx / self.N) / math.e
        S_t = 1.0 + 0.1 * t.log() * (1.0 + depth_factor)
        return S_t.unsqueeze(-1)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache cos/sin values with reciprocal blocks and power-law scaling.

        Extends parent class by applying position- and layer-dependent attention
        temperature scaling to the reciprocal block-quantized embeddings.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        b = self.block_sizes.to(device=device)
        inv_freq = self.inv_freq.to(device=device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        t_eff = torch.floor_divide(t.unsqueeze(1), b.unsqueeze(0))
        freqs = t_eff * inv_freq.unsqueeze(0)
        cos, sin = _interleave_cos_sin(freqs)
        attn_scale = self._compute_attn_scale(seq_len, device)
        self.register_buffer(
            "cos_cached",
            (cos * attn_scale).contiguous().to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (sin * attn_scale).contiguous().to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass with reciprocal block quantization and power-law attention scaling.

        Computes reciprocal block-quantized embeddings first, then applies
        position- and layer-dependent temperature scaling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Scaled reciprocal cosine values of shape (seq_len, dim).
                - sin: Scaled reciprocal sine values of shape (seq_len, dim).
        """
        if self.dynamic:
            if seq_len is None:
                seq_len = x.shape[2]
            cos, sin = super().forward(x, seq_len)
            attn_scale = self._compute_attn_scale(seq_len, x.device)
            return (cos * attn_scale).to(x.dtype), (sin * attn_scale).to(x.dtype)
        return super().forward(x, seq_len)


class LlamaFreqReciprocalScaledNoLayerRotaryEmbedding(
    LlamaFreqReciprocalRotaryEmbedding
):
    """Freq-Reciprocal Block RoPE with power-law attention scaling (no layer index).

    Inherits position encoding from LlamaFreqReciprocalRotaryEmbedding and
    applies a power-law attention temperature scaling without layer-dependent factors.
    This variant provides simpler scaling that depends only on position, making it
    suitable for scenarios where layer-specific adaptation is unnecessary.

    Attributes:
        alpha (float): Exponent for scaling factor S (currently unused, reserved).
        beta (float): Coefficient parameter (currently unused, reserved).
        Inherits all attributes from LlamaFreqReciprocalRotaryEmbedding.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device=None,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 2048,
        layer_idx: int = 0,
        num_hidden_layers: int = 32,
        dynamic: bool = False,
        alpha: float = 0.25,
        beta: float = 0.05,
    ):
        """Initialize Freq-Reciprocal RoPE with position-only attention scaling.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Extension ratio for static mode. Defaults to 1.0.
            original_max_position_embeddings (int): Model's original context length.
                Defaults to 2048.
            layer_idx (int): Index of this attention layer (unused but stored).
                Defaults to 0.
            num_hidden_layers (int): Total number of transformer layers (unused but stored).
                Defaults to 32.
            dynamic (bool): Whether to use dynamic scaling. Defaults to False.
            alpha (float): Reserved exponent parameter. Defaults to 0.25.
            beta (float): Reserved coefficient parameter. Defaults to 0.05.
        """
        super().__init__(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            device=device,
            scaling_factor=scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            layer_idx=layer_idx,
            num_hidden_layers=num_hidden_layers,
            dynamic=dynamic,
        )
        self.alpha = alpha
        self.beta = beta

    def _compute_attn_scale(self, seq_len: int, device):
        """Compute position-only power-law attention scaling factor.

        Calculates scaling using simple logarithmic growth: S_t = 1.0 + 0.1 * log(t),
        without any layer-dependent adjustment.

        Args:
            seq_len (int): Current sequence length.
            device (torch.device): Device to place result on.

        Returns:
            torch.Tensor: Attention scaling factors of shape (seq_len, 1).
        """
        t = torch.maximum(
            torch.tensor(1.0, device=device),
            torch.arange(seq_len, device=device, dtype=torch.float32)
            / self.original_max_position_embeddings,
        )
        S_t = 1.0 + 0.1 * t.log()
        return S_t.unsqueeze(-1)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache cos/sin values with reciprocal blocks and position-only scaling.

        Extends parent class by applying position-dependent (but not layer-dependent)
        attention temperature scaling to the reciprocal block-quantized embeddings.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        b = self.block_sizes.to(device=device)
        inv_freq = self.inv_freq.to(device=device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        t_eff = torch.floor_divide(t.unsqueeze(1), b.unsqueeze(0))
        freqs = t_eff * inv_freq.unsqueeze(0)
        cos, sin = _interleave_cos_sin(freqs)
        attn_scale = self._compute_attn_scale(seq_len, device)
        self.register_buffer(
            "cos_cached",
            (cos * attn_scale).contiguous().to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (sin * attn_scale).contiguous().to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass with reciprocal block quantization and position-only scaling.

        Computes reciprocal block-quantized embeddings first, then applies
        position-dependent (but layer-independent) temperature scaling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Scaled reciprocal cosine values of shape (seq_len, dim).
                - sin: Scaled reciprocal sine values of shape (seq_len, dim).
        """
        if self.dynamic:
            if seq_len is None:
                seq_len = x.shape[2]
            cos, sin = super().forward(x, seq_len)
            attn_scale = self._compute_attn_scale(seq_len, x.device)
            return (cos * attn_scale).to(x.dtype), (sin * attn_scale).to(x.dtype)
        return super().forward(x, seq_len)


class LlamaFreqReciprocalScaledAdaptiveRotaryEmbedding(
    LlamaFreqReciprocalRotaryEmbedding
):
    """Freq-Reciprocal Block RoPE with adaptive attention temperature scaling.

    Inherits position encoding from LlamaFreqReciprocalRotaryEmbedding and
    applies an adaptive attention temperature scaling that considers both
    position and dimension compression factors. The scaling adapts based on
    how far positions extend beyond the original training length, providing
    stronger compensation for more extreme extrapolation.

    Attributes:
        alpha (float): Exponent for scaling factor S (currently unused, reserved).
        beta (float): Coefficient parameter (currently unused, reserved).
        attn_scale_coef (float): Coefficient controlling scaling strength (default 0.29).
        Inherits all attributes from LlamaFreqReciprocalRotaryEmbedding.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device=None,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 2048,
        layer_idx: int = 0,
        num_hidden_layers: int = 32,
        dynamic: bool = False,
        alpha: float = 0.25,
        beta: float = 0.05,
        attn_scale_coef: float = 0.29,
    ):
        """Initialize Freq-Reciprocal RoPE with adaptive attention scaling.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Extension ratio for static mode. Defaults to 1.0.
            original_max_position_embeddings (int): Model's original context length.
                Defaults to 2048.
            layer_idx (int): Index of this attention layer. Defaults to 0.
            num_hidden_layers (int): Total number of transformer layers.
                Defaults to 32.
            dynamic (bool): Whether to use dynamic scaling. Defaults to False.
            alpha (float): Reserved exponent parameter. Defaults to 0.25.
            beta (float): Reserved coefficient parameter. Defaults to 0.05.
            attn_scale_coef (float): Adaptive scaling coefficient. Defaults to 0.29.
        """
        self.alpha = alpha
        self.beta = beta
        self.attn_scale_coef = attn_scale_coef
        super().__init__(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            device=device,
            scaling_factor=scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            layer_idx=layer_idx,
            num_hidden_layers=num_hidden_layers,
            dynamic=dynamic,
        )

    def _compute_attn_scale(self, t_eff: torch.Tensor, device):
        """Compute adaptive attention scaling factor based on effective positions.

        Calculates scaling that adapts to the degree of extrapolation beyond
        training length: mscale = 1.0 + coef * clamp((log(t_eff) - log(L_0))/log(L_0), 0).

        Args:
            t_eff (torch.Tensor): Effective (quantized) position indices.
            device (torch.device): Device to place result on.

        Returns:
            torch.Tensor: Adaptive scaling factors matching t_eff shape.
        """
        t_clipped = torch.clamp(t_eff, min=1.0)
        log_t = torch.log(t_clipped)
        log_L0 = math.log(self.original_max_position_embeddings)
        normalized_log = (log_t - log_L0) / log_L0
        clipped_log = torch.clamp(normalized_log, min=0.0)
        mscale = 1.0 + self.attn_scale_coef * clipped_log
        return mscale

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache cos/sin values with reciprocal blocks and adaptive scaling.

        Extends parent class by applying adaptive attention temperature that
        responds to the magnitude of position extrapolation.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        b = self.block_sizes.to(device=device)
        inv_freq = self.inv_freq.to(device=device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        t_eff = torch.floor_divide(t.unsqueeze(1), b.unsqueeze(0))
        freqs = t_eff * inv_freq.unsqueeze(0)
        attn_scale = self._compute_attn_scale(t_eff, device)
        cos = (freqs.cos() * attn_scale).repeat(1, 2)
        sin = (freqs.sin() * attn_scale).repeat(1, 2)
        self.register_buffer("cos_cached", cos.contiguous().to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.contiguous().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass with reciprocal block quantization and adaptive attention scaling.

        Computes reciprocal block-quantized embeddings, then applies adaptive
        scaling that strengthens with increasing extrapolation distance.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Adaptively scaled cosine values of shape (seq_len, dim).
                - sin: Adaptively scaled sine values of shape (seq_len, dim).
        """
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            S = max(1.0, seq_len / self.original_max_position_embeddings)
            if seq_len == self._dynamic_seq_len_cached and S == self._dynamic_s_cached:
                return (
                    self._dynamic_cos_cached.to(dtype),
                    self._dynamic_sin_cached.to(dtype),
                )
            b = self._compute_block_sizes(S, device=device)
            inv_freq = self.inv_freq.to(device=device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            t_eff = torch.floor_divide(t.unsqueeze(1), b.unsqueeze(0))
            freqs = t_eff * inv_freq.unsqueeze(0)
            attn_scale = self._compute_attn_scale(t_eff, device)
            cos = (freqs.cos() * attn_scale).repeat(1, 2)
            sin = (freqs.sin() * attn_scale).repeat(1, 2)
            cos = cos.to(dtype)
            sin = sin.to(dtype)
            self._dynamic_seq_len_cached = seq_len
            self._dynamic_s_cached = S
            self._dynamic_cos_cached = cos.detach()
            self._dynamic_sin_cached = sin.detach()
            return cos, sin
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            return (
                self.cos_cached[:seq_len].to(dtype),
                self.sin_cached[:seq_len].to(dtype),
            )


# ================================================================================== #
#  Dual RoPE Family                                                                  #
#                                                                                    #
#  A novel dual-position encoding approach:                                          #
#  - First half dimensions: position index = t % S (modulo operation)                #
#  - Second half dimensions: position index = t // S (integer division)              #
#                                                                                    #
#  Key design: inv_freq remains complete (size = dim // 2), only position            #
#  indices are split into two parts.                                                 #
#                                                                                    #
#  LlamaDualRoPEEmbedding         (Dual RoPE, position only)                         #
#  LlamaDualRoPEScaledEmbedding   (Dual RoPE + attention temperature)                #
# ================================================================================== #


class LlamaDualRoPEEmbedding(nn.Module):
    """Dual RoPE Embedding.

    A novel dual-position encoding approach that splits position indices into two parts
    based on the critical dimension i_star, while keeping inv_freq complete:

    - inv_freq: complete, size = dim // 2 (e.g., [f_0, f_1, f_2, ..., f_{dim//2-1}])
    - First part dimensions (i < i_star): position index = t % S (local, cyclic)
    - Second part dimensions (i >= i_star): position index = t // S (global, monotonic)

    The critical dimension i_star is computed as the number of dimensions that complete
    at least one full rotation within the original context window::
        r_i = L_0 * θ_i / (2π),  i_star = first index where r_i < 1

    This design allows:
    1. High-frequency dimensions (i < i_star): capture local position via modulo
    2. Low-frequency dimensions (i >= i_star): capture global position via integer division

    Attributes:
        dim (int): Dimension of the embedding (head dimension).
        max_position_embeddings (int): Maximum sequence length for caching.
        base (int): Base frequency for computing inverse frequencies.
        scaling_factor (float): Scaling factor for position interpolation.
        S (int): Effective scaling factor (max of scaling_factor and original_max).
        original_max_position_embeddings (int): Original context window size.
        dynamic (bool): Whether to dynamically recompute frequencies.
        i_star (int): Critical dimension index for splitting position indices.
        inv_freq (torch.Tensor): Complete inverse frequency buffer.
        inv_freq_1 (torch.Tensor): High-frequency inverse frequencies (i < i_star).
        inv_freq_2 (torch.Tensor): Low-frequency inverse frequencies (i >= i_star).
        cos_cached (torch.Tensor): Cached cosine values (static mode only).
        sin_cached (torch.Tensor): Cached sine values (static mode only).
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
    ):
        """Initialize Dual RoPE Embedding.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Scaling factor for position interpolation.
                Defaults to 1.0.
            original_max_position_embeddings (int): Original context window size.
                Defaults to 2048.
            dynamic (bool): Whether to dynamically recompute frequencies.
                Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = max(1.0, scaling_factor)
        self.S = max(self.scaling_factor, original_max_position_embeddings)
        self.original_max_position_embeddings = original_max_position_embeddings
        self.dynamic = dynamic

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        r = original_max_position_embeddings * inv_freq / (2.0 * math.pi)
        i_star = int((r >= 1.0).sum().item())
        self.i_star = max(1, min(i_star, dim // 2 - 1))

        self.register_buffer("inv_freq_1", inv_freq[: self.i_star], persistent=False)
        self.register_buffer("inv_freq_2", inv_freq[self.i_star :], persistent=False)

        self._dynamic_seq_len_cached = -1
        self._dynamic_S_cached = -1
        self._dynamic_cos_cached = None
        self._dynamic_sin_cached = None

        if not dynamic:
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=self.inv_freq.device,
                dtype=torch.get_default_dtype(),
            )

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache cos/sin values with dual position encoding.

        Splits position indices into cyclic (mod S) and monotonic (div S) parts,
        applies them to respective frequency halves, then concatenates results.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        S = self.S
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        divmod_result = torch.div(t, S, rounding_mode="floor")
        pos_1 = t - divmod_result * S
        pos_2 = divmod_result
        self.register_buffer("pos_1_cached", pos_1, persistent=False)
        self.register_buffer("pos_2_cached", pos_2, persistent=False)
        inv_freq_1 = self.inv_freq_1.to(device=device)
        inv_freq_2 = self.inv_freq_2.to(device=device)
        freqs_1 = pos_1.unsqueeze(1) * inv_freq_1.unsqueeze(0)
        freqs_2 = pos_2.unsqueeze(1) * inv_freq_2.unsqueeze(0)
        freqs = torch.cat([freqs_1, freqs_2], dim=-1)
        cos = freqs.cos().repeat(1, 2)
        sin = freqs.sin().repeat(1, 2)
        self.register_buffer("cos_cached", cos.contiguous().to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.contiguous().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass to retrieve or compute dual RoPE embeddings.

        Applies dual position encoding with cyclic local positions for high
        frequencies and monotonic global positions for low frequencies.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Dual-encoded cosine values of shape (seq_len, dim).
                - sin: Dual-encoded sine values of shape (seq_len, dim).
        """
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            scaling_factor = max(1.0, seq_len / self.original_max_position_embeddings)
            S = max(scaling_factor, self.original_max_position_embeddings)
            if seq_len == self._dynamic_seq_len_cached and S == self._dynamic_S_cached:
                return (
                    self._dynamic_cos_cached.to(dtype),
                    self._dynamic_sin_cached.to(dtype),
                )
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            divmod_result = torch.div(t, S, rounding_mode="floor")
            pos_1 = t - divmod_result * S
            pos_2 = divmod_result
            inv_freq_1 = self.inv_freq_1.to(device=device)
            inv_freq_2 = self.inv_freq_2.to(device=device)
            freqs_1 = pos_1.unsqueeze(1) * inv_freq_1.unsqueeze(0)
            freqs_2 = pos_2.unsqueeze(1) * inv_freq_2.unsqueeze(0)
            freqs = torch.cat([freqs_1, freqs_2], dim=-1)
            cos = freqs.cos().repeat(1, 2).to(dtype)
            sin = freqs.sin().repeat(1, 2).to(dtype)
            self._dynamic_seq_len_cached = seq_len
            self._dynamic_S_cached = S
            self._dynamic_cos_cached = cos.detach()
            self._dynamic_sin_cached = sin.detach()
            return cos, sin
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            return (
                self.cos_cached[:seq_len].to(dtype),
                self.sin_cached[:seq_len].to(dtype),
            )


class LlamaDualRoPEScaledEmbedding(LlamaDualRoPEEmbedding):
    """Dual RoPE Embedding with attention temperature scaling.

    Inherits the dual-position encoding from LlamaDualRoPEEmbedding and
    adds adaptive attention temperature scaling for better long-context handling.

    The attention scaling uses the entropy-based formula: mscale = max(1.0, log(t)/log(L_0))
    where L_0 is the original_max_position_embeddings and t is the position index.
    This scaling does not distinguish between dimensions, providing uniform compensation.

    Attributes:
        Inherits all attributes from LlamaDualRoPEEmbedding.
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
        attn_scale_coef: float = 0.1,
    ):
        """Initialize Dual RoPE Embedding with attention temperature scaling.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Scaling factor for position interpolation.
                Defaults to 1.0.
            original_max_position_embeddings (int): Original context window size.
                Defaults to 2048.
            dynamic (bool): Whether to dynamically recompute frequencies.
                Defaults to False.
            attn_scale_coef (float): Coefficient for attention scaling (currently unused,
                reserved for future variants). Defaults to 0.1.
        """
        super().__init__(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            device=device,
            scaling_factor=scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            dynamic=dynamic,
        )

    def _compute_attn_scale(self, seq_len: int, device):
        """Compute entropy-based attention scaling factor.

        Calculates uniform scaling using logarithmic ratio to original context length:
        mscale = max(1.0, log(t+1) / log(L_0)), applied identically to all positions.

        Args:
            seq_len (int): Current sequence length.
            device (torch.device): Device to place result on.

        Returns:
            torch.Tensor: Attention scaling factors of shape (seq_len, 1).
        """
        t = torch.arange(seq_len, device=device, dtype=torch.float32) + 1.0
        log_t = torch.log(t)
        log_L0 = math.log(self.original_max_position_embeddings)
        mscale = torch.maximum(
            torch.tensor(1.0, device=device),
            log_t / log_L0,
        )
        return mscale.unsqueeze(-1)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache cos/sin values with dual encoding and attention scaling.

        Extends parent class by applying entropy-based attention temperature
        to the dual-encoded embeddings.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        S = self.S
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        divmod_result = torch.div(t, S, rounding_mode="floor")
        pos_1 = t - divmod_result * S
        pos_2 = divmod_result
        inv_freq_1 = self.inv_freq_1.to(device=device)
        inv_freq_2 = self.inv_freq_2.to(device=device)
        freqs_1 = pos_1.unsqueeze(1) * inv_freq_1.unsqueeze(0)
        freqs_2 = pos_2.unsqueeze(1) * inv_freq_2.unsqueeze(0)
        freqs = torch.cat([freqs_1, freqs_2], dim=-1)
        cos = freqs.cos().repeat(1, 2)
        sin = freqs.sin().repeat(1, 2)
        attn_scale = self._compute_attn_scale(seq_len, device)
        self.register_buffer(
            "cos_cached",
            (cos * attn_scale).contiguous().to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (sin * attn_scale).contiguous().to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass with dual position encoding and entropy-based attention scaling.

        Computes dual-encoded embeddings first, then applies uniform logarithmic
        attention temperature scaling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Scaled dual cosine values of shape (seq_len, dim).
                - sin: Scaled dual sine values of shape (seq_len, dim).
        """
        if self.dynamic:
            if seq_len is None:
                seq_len = x.shape[2]
            scaling_factor = max(1.0, seq_len / self.original_max_position_embeddings)
            S = max(scaling_factor, self.original_max_position_embeddings)
            if seq_len == self._dynamic_seq_len_cached and S == self._dynamic_S_cached:
                cached_cos = self._dynamic_cos_cached.to(x.dtype)
                cached_sin = self._dynamic_sin_cached.to(x.dtype)
                attn_scale = self._compute_attn_scale(seq_len, x.device)
                return (
                    (cached_cos * attn_scale).to(x.dtype),
                    (cached_sin * attn_scale).to(x.dtype),
                )
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            divmod_result = torch.div(t, S, rounding_mode="floor")
            pos_1 = t - divmod_result * S
            pos_2 = divmod_result
            inv_freq_1 = self.inv_freq_1.to(device=x.device)
            inv_freq_2 = self.inv_freq_2.to(device=x.device)
            freqs_1 = pos_1.unsqueeze(1) * inv_freq_1.unsqueeze(0)
            freqs_2 = pos_2.unsqueeze(1) * inv_freq_2.unsqueeze(0)
            freqs = torch.cat([freqs_1, freqs_2], dim=-1)
            cos = freqs.cos().repeat(1, 2)
            sin = freqs.sin().repeat(1, 2)
            attn_scale = self._compute_attn_scale(seq_len, x.device)
            cos = (cos * attn_scale).to(x.dtype)
            sin = (sin * attn_scale).to(x.dtype)
            self._dynamic_seq_len_cached = seq_len
            self._dynamic_S_cached = S
            self._dynamic_cos_cached = cos.detach()
            self._dynamic_sin_cached = sin.detach()
            return cos, sin
        return super().forward(x, seq_len)


# ================================================================================== #
#  Inverse Dual RoPE family                                                          #
#                                                                                    #
#  LlamaInverseDualRoPEEmbedding       (Inverse-Dual-RoPE, position only)            #
#  LlamaInverseDualRoPEScaledEmbedding (Inverse-Dual-RoPE + attention temperature)   #
# ================================================================================== #


class LlamaInverseDualRoPEEmbedding(nn.Module):
    """Inverse Dual RoPE Embedding.Normal name: BiSpaceRoPE.

    A novel inverse dual-position encoding approach that splits position indices into
    two parts based on the critical dimension i_star, while keeping inv_freq complete.
    This is the INVERSE of LlamaDualRoPEEmbedding — the high/low frequency operations
    are swapped:

    - inv_freq: complete, size = dim // 2 (e.g., [f_0, f_1, f_2, ..., f_{dim//2-1}])
    - High-frequency dimensions (i < i_star): position index = t (global, monotonic)
    - Low-frequency dimensions (i >= i_star): position index = t % L_0 (local, cyclic)

    The critical dimension i_star is computed as the number of dimensions that complete
    at least one full rotation within the original context window::
        r_i = L_0 * θ_i / (2π),  i_star = first index where r_i < 1

    Key difference from Dual-RoPE: denominator uses L_0 (original context length),
    not S (scaled context length). This makes the cyclic component always relative to
    the training distribution.

    Attributes:
        dim (int): Dimension of the embedding (head dimension).
        max_position_embeddings (int): Maximum sequence length for caching.
        base (int): Base frequency for computing inverse frequencies.
        scaling_factor (float): Scaling factor for position interpolation.
        original_max_position_embeddings (int): Original context window size.
        dynamic (bool): Whether to dynamically recompute frequencies (currently unused).
        i_star (int): Critical dimension index for splitting position indices.
        inv_freq (torch.Tensor): Complete inverse frequency buffer.
        inv_freq_1 (torch.Tensor): High-frequency inverse frequencies (i < i_star).
        inv_freq_2 (torch.Tensor): Low-frequency inverse frequencies (i >= i_star).
        cos_cached (torch.Tensor): Cached cosine values.
        sin_cached (torch.Tensor): Cached sine values.
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
    ):
        """Initialize Inverse Dual RoPE Embedding.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Scaling factor (currently unused, reserved).
                Defaults to 1.0.
            original_max_position_embeddings (int): Original context window size.
                Defaults to 2048.
            dynamic (bool): Whether to dynamically recompute frequencies (unused).
                Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = max(1.0, scaling_factor)
        self.original_max_position_embeddings = original_max_position_embeddings
        self.dynamic = dynamic

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        r = original_max_position_embeddings * inv_freq / (2.0 * math.pi)
        i_star = int((r >= 1.0).sum().item())
        self.i_star = max(1, min(i_star, dim // 2 - 1))

        self.register_buffer("inv_freq_1", inv_freq[: self.i_star], persistent=False)
        self.register_buffer("inv_freq_2", inv_freq[self.i_star :], persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache cos/sin values with inverse dual position encoding.

        Swaps the assignment compared to Dual-RoPE: high frequencies get raw position
        indices (t, monotonic) and low frequencies get cyclic positions (mod L_0).

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        L_0 = self.original_max_position_embeddings
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        pos_1 = t
        pos_2 = t % L_0
        self.register_buffer("pos_1_cached", pos_1, persistent=False)
        self.register_buffer("pos_2_cached", pos_2, persistent=False)
        inv_freq_1 = self.inv_freq_1.to(device=device)
        inv_freq_2 = self.inv_freq_2.to(device=device)
        freqs_1 = pos_1.unsqueeze(1) * inv_freq_1.unsqueeze(0)
        freqs_2 = pos_2.unsqueeze(1) * inv_freq_2.unsqueeze(0)
        freqs = torch.cat([freqs_1, freqs_2], dim=-1)
        cos = freqs.cos().repeat(1, 2)
        sin = freqs.sin().repeat(1, 2)
        self.register_buffer("cos_cached", cos.contiguous().to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.contiguous().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass to retrieve inverse dual RoPE embeddings.

        Returns cached embeddings, recomputing only if sequence length exceeds cache.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Inverse dual cosine values of shape (seq_len, dim).
                - sin: Inverse dual sine values of shape (seq_len, dim).
        """
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
        return (
            self.cos_cached[:seq_len].to(dtype),
            self.sin_cached[:seq_len].to(dtype),
        )


class LlamaInverseDualRoPEScaledEmbedding(LlamaInverseDualRoPEEmbedding):
    """Inverse Dual RoPE Embedding with piecewise global-local attention scaling.

    Inherits the inverse dual-position encoding from LlamaInverseDualRoPEEmbedding and
    adds a decomposed scaling function s(t) = global(t) × local(t) for better
    long-context handling.

    Design idea: Decompose the scaling function s(t) into global term × local term:

        s(t) = 1                              t < L_0
        s(t) = (1 + α·ln(k+1)) · (1 + β·e^(-γr))   t ≥ L_0

    where:
        k = ⌊t / L_0⌋ ≥ 1          (segment index, global)
        r = (t mod L_0) / L_0 ∈ [0, 1]  (intra-segment position, local)

    Parameter meanings:
        α : Global term growth rate with segment index (YaRN-like ln growth). Default: 0.1
        β : Jump compensation amplitude at segment boundaries. Default: 0.5
        γ : Intra-segment decay rate for local compensation. Default: 2.0

    Attributes:
        Inherits all attributes from LlamaInverseDualRoPEEmbedding.
        alpha (float): Global term growth rate coefficient.
        beta (float): Boundary jump compensation amplitude.
        gamma (float): Intra-segment exponential decay rate.
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
        beta: float = 0.5,
        gamma: float = 2.0,
    ):
        """Initialize Inverse Dual RoPE Embedding with global-local scaling.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_position_embeddings (int): Maximum sequence length for caching.
                Defaults to 2048.
            base (int): Base frequency. Defaults to 10000.
            device (torch.device): Device to place tensors on. Defaults to None.
            scaling_factor (float): Scaling factor (currently unused, reserved).
                Defaults to 1.0.
            original_max_position_embeddings (int): Original context window size L_0.
                Defaults to 2048.
            dynamic (bool): Whether to dynamically recompute frequencies (unused).
                Defaults to False.
            alpha (float): Global term growth rate (YaRN-like ln growth).
                Controls how fast the global scale grows with segment index k.
                Defaults to 0.1.
            beta (float): Boundary jump compensation amplitude.
                Controls the magnitude of local compensation at segment boundaries.
                Defaults to 0.5.
            gamma (float): Intra-segment decay rate.
                Controls how quickly the local compensation decays within each segment.
                Defaults to 2.0.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        super().__init__(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            device=device,
            scaling_factor=scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            dynamic=dynamic,
        )

    def _compute_attn_scale(self, seq_len: int, device):
        """Compute global-local decomposed attention scaling factor.

        Implements the piecewise scaling function:
            s(t) = 1                                    t < L_0
            s(t) = (1 + α·ln(k+1)) · (1 + β·e^(-γr))   t ≥ L_0

        where k = floor(t / L_0) is the segment index (global coordinate),
        and r = (t mod L_0) / L_0 is the normalized intra-segment position (local coordinate).

        The global term (1 + α·ln(k+1)) provides monotonic YaRN-like logarithmic growth
        across segments, compensating for cumulative attention entropy loss.

        The local term (1 + β·e^(-γr)) provides a jump-up compensation at each segment
        boundary (r ≈ 0) that decays exponentially within the segment (r → 1),
        addressing the discontinuity issue at segment transitions.

        Args:
            seq_len (int): Current sequence length.
            device (torch.device): Device to place result on.

        Returns:
            torch.Tensor: Attention scaling factors of shape (seq_len, 1).
        """
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        L_0 = self.original_max_position_embeddings

        mask = t < L_0
        k = torch.floor(t / L_0).clamp(min=1.0)
        r = (t % L_0) / L_0

        global_term = 1.0 + self.alpha * torch.log(k + 1.0)
        local_term = 1.0 + self.beta * torch.exp(-self.gamma * r)

        s_t = torch.where(
            mask,
            torch.tensor(1.0, device=device),
            global_term * local_term,
        )
        return s_t.unsqueeze(-1)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Compute and cache cos/sin values with inverse dual encoding and global-local scaling.

        Extends parent class by applying the decomposed scaling function s(t) =
        global(t) × local(t) to the inverse dual-encoded embeddings, where the global
        term captures cross-segment accumulation and the local term handles intra-segment
        boundary compensation.

        Args:
            seq_len (int): Sequence length to cache embeddings for.
            device (torch.device): Device to place cached tensors on.
            dtype (torch.dtype): Data type for cached tensors.

        Returns:
            None
        """
        self.max_seq_len_cached = seq_len
        L_0 = self.original_max_position_embeddings
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        pos_1 = t
        pos_2 = t % L_0
        inv_freq_1 = self.inv_freq_1.to(device=device)
        inv_freq_2 = self.inv_freq_2.to(device=device)
        freqs_1 = pos_1.unsqueeze(1) * inv_freq_1.unsqueeze(0)
        freqs_2 = pos_2.unsqueeze(1) * inv_freq_2.unsqueeze(0)
        freqs = torch.cat([freqs_1, freqs_2], dim=-1)
        cos = freqs.cos().repeat(1, 2)
        sin = freqs.sin().repeat(1, 2)
        attn_scale = self._compute_attn_scale(seq_len, device)
        self.register_buffer(
            "cos_cached",
            (cos * attn_scale).contiguous().to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (sin * attn_scale).contiguous().to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """Forward pass with inverse dual position encoding and global-local scaling.

        Returns cached embeddings with the decomposed attention temperature applied.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            seq_len (int, optional): Requested sequence length. If None, inferred
                from input. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cos: Scaled inverse dual cosine values of shape (seq_len, dim).
                - sin: Scaled inverse dual sine values of shape (seq_len, dim).
        """
        if seq_len is None:
            seq_len = x.shape[2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )
