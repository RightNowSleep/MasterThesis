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


class LlamaRotaryEmbedding(nn.Module):
    """
    Standard Llama Rotary Position Embedding (RoPE).

    Implements the original RoPE as described in the Llama paper. Computes
    cos/sin caches for rotary position embeddings with fixed base frequency.

    Attributes:
        dim (int): Dimension of the embedding (head dimension).
        max_position_embeddings (int): Maximum sequence length for caching.
        base (int): Base frequency for computing inverse frequencies.
        inv_freq (torch.Tensor): Inverse frequency buffer.
        cos_cached (torch.Tensor): Cached cosine values.
        sin_cached (torch.Tensor): Cached sine values.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
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
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached,
            device=device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
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
    """
    Position Interpolation (PI) with linear position scaling.

    Implements linear position scaling where positions are divided by the
    scaling factor to extend the context window. Supports both static and
    dynamic scaling modes.

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

        if not dynamic:
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=self.inv_freq.device,
                dtype=torch.get_default_dtype(),
            )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if self.dynamic:
            s = (
                max(seq_len, self.original_max_position_embeddings)
                / self.original_max_position_embeddings
            )
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            t = t / s
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)
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
    """
    NTK-aware RoPE scaling with base frequency modification.

    Modifies the RoPE base frequency to extend context length. The base is
    scaled as base' = base * s^(d/(d-2)) where s is the scaling factor.

    Attributes:
        dim (int): Dimension of the embedding.
        max_position_embeddings (int): Maximum sequence length for caching.
        original_max_position_embeddings (int): Model's original context length.
        base (int): Original base frequency.
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
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.dynamic = dynamic

        if not dynamic:
            modified_base = base * scaling_factor ** (dim / (dim - 2))
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
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if self.dynamic:
            s = max(1.0, seq_len / self.original_max_position_embeddings)
            modified_base = self.base * s ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                modified_base
                ** (
                    torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32)
                    / self.dim
                )
            )
            t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)
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
    """
    NTK-by-parts RoPE scaling with piecewise frequency blending.

    Splits dimensions into three groups based on rotation frequency within
    the original context: high-frequency (unchanged), low-frequency (linearly
    interpolated), and transition region (smooth blend).

    Attributes:
        dim (int): Dimension of the embedding.
        max_position_embeddings (int): Maximum sequence length for caching.
        original_max_position_embeddings (int): Model's original context length.
        base (int): Base frequency for computing inverse frequencies.
        scaling_factor (float): Extension ratio for static mode.
        alpha_ntk (float): Lower boundary for frequency blending.
        beta_ntk (float): Upper boundary for frequency blending.
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
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.alpha_ntk = alpha
        self.beta_ntk = beta
        self.dynamic = dynamic

        if not dynamic:
            inv_freq = self._compute_inv_freq(scaling_factor, device=device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=self.inv_freq.device,
                dtype=torch.get_default_dtype(),
            )

    def _compute_inv_freq(self, scaling_factor, device=None):
        theta_d = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )
        lambda_d = 2.0 * math.pi / theta_d
        r_d = self.original_max_position_embeddings / lambda_d
        w_ext = torch.clamp(
            (r_d - self.alpha_ntk) / (self.beta_ntk - self.alpha_ntk),
            0.0,
            1.0,
        )
        return theta_d * w_ext + (1.0 - w_ext) * theta_d / scaling_factor

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if self.dynamic:
            s = max(1.0, seq_len / self.original_max_position_embeddings)
            inv_freq = self._compute_inv_freq(s, device=x.device)
            t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)
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
    """
    YaRN (Yet Another RoPE extensioN) with attention temperature scaling.

    Combines NTK-by-parts frequency blending with attention temperature
    scaling (t = 1 + 0.1 * ln(s)) to improve extrapolation performance.

    Attributes:
        attention_scaling (float): Temperature scaling factor (static mode).
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
        super()._set_cos_sin_cache(seq_len, device, dtype)
        if self.attention_scaling is not None:
            self.register_buffer(
                "cos_cached",
                self.cos_cached * self.attention_scaling,
                persistent=False,
            )
            self.register_buffer(
                "sin_cached",
                self.sin_cached * self.attention_scaling,
                persistent=False,
            )

    def forward(self, x, seq_len=None):
        if self.dynamic:
            s = max(1.0, seq_len / self.original_max_position_embeddings)
            attention_scaling = 1.0 + 0.1 * math.log(s)
            inv_freq = self._compute_inv_freq(s, device=x.device)
            t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return (
                emb.cos().to(dtype=x.dtype) * attention_scaling,
                emb.sin().to(dtype=x.dtype) * attention_scaling,
            )
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
    """
    Compute layer-dependent attention amplitude scalar with inverted-U profile.

    Middle layers receive weaker correction (u_norm approaches 1, layer_alpha approaches 0),
    while first and last layers receive stronger correction (u_norm approaches 0, layer_alpha approaches 0.1).

    Args:
        layer_idx (int): Index of the current attention layer.
        num_hidden_layers (int): Total number of transformer layers.
        seq_len (int): Current sequence length.
        original_max_position_embeddings (int): Model's original context length.

    Returns:
        float: Python float suitable for multiplication with cos/sin tensors.
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
    """
    Layer-aware My RoPE with position encoding only.

    Implements NTK-by-parts frequency blending with layer-adaptive alpha/beta
    boundaries using an inverted-U profile across layers. Does not apply
    attention temperature correction.

    Attributes:
        dim (int): Dimension of the embedding.
        max_position_embeddings (int): Maximum sequence length for caching.
        base (int): Base frequency for computing inverse frequencies.
        scaling_factor (float): Extension ratio for static mode.
        N (int): Total number of transformer layers.
        original_max_position_embeddings (int): Model's original context length.
        layer_idx (int): Index of this attention layer.
        alpha (float): Alpha parameter for layer adaptation.
        dynamic (bool): Whether to use dynamic scaling.
        inv_freq (torch.Tensor): Inverse frequency buffer (static mode).
        inv_freq_base (torch.Tensor): Base inverse frequency (dynamic mode).
        w_ext (torch.Tensor): Blending weights (dynamic mode).
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

        if not dynamic:
            lambda_d = 2 * math.pi / theta_d
            r_d = original_max_position_embeddings / lambda_d
            a, b = 1.0, 32.0
            w_ext = torch.clamp((r_d - a) / (b - a), 0.0, 1.0)

            # Inverted-U layer influence (identical formula to dynamic _build_w_ext)
            layer_norm = 2.0 * layer_idx / (num_hidden_layers - 1) - 1.0
            u_norm = 1.0 - layer_norm**2
            alpha_eff = 1.0 + 1.0 * u_norm
            beta_eff = 32.0 + 8.0 * u_norm
            w_ext_layer = torch.clamp(
                (r_d - alpha_eff) / (beta_eff - alpha_eff),
                0.0,
                1.0,
            )

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
            self._build_w_ext()

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached",
            emb.cos().to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            emb.sin().to(dtype),
            persistent=False,
        )

    def _build_w_ext(self):
        """Pre-compute the per-dimension blending mask with layer-adaptive parameters."""
        theta_d = self.inv_freq_base
        lambda_d = 2 * math.pi / theta_d
        r_d = self.original_max_position_embeddings / lambda_d

        layer_norm = 2.0 * self.layer_idx / (self.N - 1) - 1.0
        u_norm = 1.0 - layer_norm**2
        alpha_eff = 1.0 + 1.0 * u_norm
        beta_eff = 32.0 + 8.0 * u_norm
        w_ext_layer = torch.clamp(
            (r_d - alpha_eff) / (beta_eff - alpha_eff),
            0.0,
            1.0,
        )
        self.register_buffer("w_ext", w_ext_layer, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if self.dynamic:
            device, dtype = x.device, x.dtype
            S = (
                max(seq_len, self.original_max_position_embeddings)
                / self.original_max_position_embeddings
            )
            inv_freq = (
                self.w_ext * self.inv_freq_base
                + (1.0 - self.w_ext) * self.inv_freq_base / S
            )
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=x.dtype),
                self.sin_cached[:seq_len].to(dtype=x.dtype),
            )


# ---------------------------------------------------------------------------- #


class LlamaMyScaledRotaryEmbedding(LlamaMyRotaryEmbedding):
    """
    Layer-aware My RoPE with attention temperature scaling.

    Inherits position encoding from LlamaMyRotaryEmbedding and applies a
    layer-dependent attention temperature scalar with inverted-U profile.

    Attributes:
        Inherits all attributes from LlamaMyRotaryEmbedding.
    """

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Build unscaled cache via parent, then multiply in the attention scalar.
        super()._set_cos_sin_cache(seq_len, device, dtype)
        attn_scale = _layer_aware_attn_scale(
            self.layer_idx,
            self.N,
            seq_len,
            self.original_max_position_embeddings,
        )
        self.register_buffer(
            "cos_cached",
            (self.cos_cached * attn_scale).to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (self.sin_cached * attn_scale).to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if self.dynamic:
            # Delegate position encoding to base, then scale.
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
        # Static: cache already carries the attention scalar (applied in
        # _set_cos_sin_cache above), so just return from cache normally.
        return super().forward(x, seq_len)


# ---------------------------------------------------------------------------- #


class LlamaMyRotaryEmbedding2(nn.Module):
    """
    Multi-scale My RoPE with position encoding only.

    Splits the head dimension into three sub-spaces for local, paragraph, and
    document scales, each with its own NTK-by-parts parameters and base frequency.
    Does not apply attention temperature correction.

    Attributes:
        dim (int): Dimension of the embedding.
        base (int): Base frequency for computing inverse frequencies.
        max_position_embeddings (int): Maximum sequence length for caching.
        original_max_position_embeddings (int): Model's original context length.
        scaling_factor (float): Extension ratio for static mode.
        alpha (float): Alpha parameter for scaling.
        dynamic (bool): Whether to use dynamic scaling.
        layer_idx (int): Index of this attention layer.
        N (int): Total number of transformer layers.
        scales (list): Configuration for each sub-space scale.
        scale_buffers (list): Buffers for each sub-space.
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

        # ── Build per-scale inv_freq buffers (shared by both modes) ──── #
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
                scale_dim = dim - current_dim  # absorb rounding remainder
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

        if not dynamic:
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=device or self.scale_buffers[0]["inv_freq"].device,
                dtype=torch.get_default_dtype(),
            )

    def _get_scale_inv_freq_static(self, scale_idx: int, device):
        """
        Compute blended inv_freq for one sub-space using fixed scaling_factor.

        Args:
            scale_idx (int): Index of the sub-space scale.
            device: Device for tensor operations.

        Returns:
            torch.Tensor: Blended inverse frequencies for the sub-space.
        """
        buffer = self.scale_buffers[scale_idx]
        theta_d = buffer["inv_freq"].to(device=device)
        window = buffer["window"]

        if scale_idx == 0:
            ntk_alpha, ntk_beta = 0.8, 24.0
        elif scale_idx == 1:
            ntk_alpha, ntk_beta = 1.0, 32.0
        else:
            ntk_alpha, ntk_beta = 1.2, 40.0

        lambda_d = 2 * math.pi / theta_d
        r_d = window / lambda_d
        w_ext = torch.clamp((r_d - ntk_alpha) / (ntk_beta - ntk_alpha), 0.0, 1.0)
        return w_ext * theta_d + (1.0 - w_ext) * theta_d / self.scaling_factor

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        cos_final = torch.zeros(seq_len, self.dim, device=device)
        sin_final = torch.zeros(seq_len, self.dim, device=device)
        for i, buffer in enumerate(self.scale_buffers):
            scale_dim = buffer["dim_end"] - buffer["dim_start"]
            if scale_dim <= 0:
                continue
            inv_freq_scaled = self._get_scale_inv_freq_static(i, device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq_scaled.dtype)
            freqs = torch.outer(t, inv_freq_scaled)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos_final[:, buffer["dim_start"] : buffer["dim_end"]] = emb.cos()
            sin_final[:, buffer["dim_start"] : buffer["dim_end"]] = emb.sin()

        self.register_buffer("cos_cached", cos_final.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin_final.to(dtype), persistent=False)

    def _get_scale_inv_freq_dynamic(self, scale_idx: int, seq_len: int, device):
        """
        Compute blended inv_freq for one sub-space using runtime seq_len.

        Args:
            scale_idx (int): Index of the sub-space scale.
            seq_len (int): Current sequence length.
            device: Device for tensor operations.

        Returns:
            torch.Tensor: Blended inverse frequencies for the sub-space.
        """
        buffer = self.scale_buffers[scale_idx]
        theta_d = buffer["inv_freq"]
        window = buffer["window"]

        if scale_idx == 0:
            ntk_alpha, ntk_beta = 0.8, 24.0
            s = max(1.0, seq_len / window)
        elif scale_idx == 1:
            ntk_alpha, ntk_beta = 1.0, 32.0
            s = max(1.0, seq_len / window)
        else:
            ntk_alpha, ntk_beta = 1.2, 40.0
            s = max(1.0, seq_len / window)

        lambda_d = 2 * math.pi / theta_d
        r_d = window / lambda_d
        w_ext = torch.clamp((r_d - ntk_alpha) / (ntk_beta - ntk_alpha), 0.0, 1.0)
        return (w_ext * theta_d + (1.0 - w_ext) * theta_d / s).to(device=device)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            cos_final = torch.zeros(seq_len, self.dim, device=device, dtype=dtype)
            sin_final = torch.zeros(seq_len, self.dim, device=device, dtype=dtype)
            for i, buffer in enumerate(self.scale_buffers):
                scale_dim = buffer["dim_end"] - buffer["dim_start"]
                if scale_dim <= 0:
                    continue
                inv_freq_scaled = self._get_scale_inv_freq_dynamic(i, seq_len, device)
                t = torch.arange(seq_len, device=device, dtype=inv_freq_scaled.dtype)
                freqs = torch.outer(t, inv_freq_scaled)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos_final[:, buffer["dim_start"] : buffer["dim_end"]] = emb.cos()
                sin_final[:, buffer["dim_start"] : buffer["dim_end"]] = emb.sin()
            return cos_final.to(dtype=dtype), sin_final.to(dtype=dtype)
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=dtype),
                self.sin_cached[:seq_len].to(dtype=dtype),
            )


# ---------------------------------------------------------------------------- #


class LlamaMyScaledRotaryEmbedding2(LlamaMyRotaryEmbedding2):
    """
    Multi-scale My RoPE 2 with attention temperature scaling.

    Inherits position encoding from LlamaMyRotaryEmbedding2 and applies the
    same layer-dependent attention temperature scalar.

    Attributes:
        Inherits all attributes from LlamaMyRotaryEmbedding2.
    """

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        super()._set_cos_sin_cache(seq_len, device, dtype)
        attn_scale = _layer_aware_attn_scale(
            self.layer_idx,
            self.N,
            seq_len,
            self.original_max_position_embeddings,
        )
        self.register_buffer(
            "cos_cached",
            (self.cos_cached * attn_scale).to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (self.sin_cached * attn_scale).to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
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
    """
    Block-Layered RoPE with position encoding only.

    Implements quantized effective position indices using per-dimension block
    sizes that grow exponentially with dimension index. Prevents angular value
    out-of-distribution when extending beyond original context length.

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
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scaling_factor = scaling_factor
        self.layer_idx = layer_idx
        self.N = num_hidden_layers
        self.dynamic = dynamic

        # ── Base inv_freq (unscaled, shared by both modes) ───────────── #
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # ── Critical dimension i* ────────────────────────────────────── #
        # r_i = L_0 / λ_i = L_0 * θ_i / (2π)
        # i* = number of dimensions with r_i >= 1  (those that complete
        #      at least one full rotation inside the original context)
        r_d = original_max_position_embeddings * inv_freq / (2.0 * math.pi)
        i_star = int((r_d >= 1.0).sum().item())
        # Guard against edge cases (all dims or no dims complete a cycle)
        self.i_star = max(1, min(i_star, dim // 2 - 1))

        if not dynamic:
            block_sizes = self._compute_block_sizes(scaling_factor, device=device)
            self.register_buffer("block_sizes", block_sizes, persistent=False)
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=device or inv_freq.device,
                dtype=torch.get_default_dtype(),
            )
        # dynamic: block_sizes recomputed each forward pass

    # ------------------------------------------------------------------ #

    def _compute_block_sizes(self, S: float, device=None) -> torch.Tensor:
        """
        Compute per-dimension block sizes for extension ratio S.

        Block sizes grow exponentially: b_i = clamp(S^(i / i*), 1, S) for i < i*,
        and b_i = S for i >= i*. When S = 1.0, all b_i = 1.0 (standard RoPE).

        Args:
            S (float): Extension ratio for scaling.
            device: Device for tensor creation.

        Returns:
            torch.Tensor: Float32 tensor of shape [dim//2] containing block sizes.
        """
        half_dim = self.dim // 2
        indices = torch.arange(half_dim, device=device, dtype=torch.float32)
        # exponent: 0 at i=0, 1 at i=i_star, clamped to 1 beyond
        exponent = torch.clamp(indices / float(self.i_star), 0.0, 1.0)
        b = torch.clamp(float(S) ** exponent, min=1.0, max=float(S))
        return b

    # ------------------------------------------------------------------ #

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        b = self.block_sizes.to(device=device)  # [d//2]
        inv_freq = self.inv_freq.to(device=device)  # [d//2]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)  # [N]

        # t_eff[n, i] = floor(t[n] / b[i])  →  [N, d//2]
        t_eff = torch.floor(t[:, None] / b[None, :])
        freqs = t_eff * inv_freq[None, :]  # [N, d//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [N, d]

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, seq_len: int = None):
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            S = max(1.0, seq_len / self.original_max_position_embeddings)
            b = self._compute_block_sizes(S, device=device)  # [d//2]
            inv_freq = self.inv_freq.to(device=device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            t_eff = torch.floor(t[:, None] / b[None, :])  # [N, d//2]
            freqs = t_eff * inv_freq[None, :]  # [N, d//2]
            emb = torch.cat((freqs, freqs), dim=-1)  # [N, d]
            return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=dtype),
                self.sin_cached[:seq_len].to(dtype=dtype),
            )


# ---------------------------------------------------------------------------- #


class LlamaBlockLayeredScaledRotaryEmbedding(LlamaBlockLayeredRotaryEmbedding):
    """
    Block-Layered RoPE with attention temperature scaling.

    Inherits position encoding from LlamaBlockLayeredRotaryEmbedding and
    applies a layer-dependent attention temperature scalar with inverted-U profile.

    Attributes:
        Inherits all attributes from LlamaBlockLayeredRotaryEmbedding.
    """

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        super()._set_cos_sin_cache(seq_len, device, dtype)
        attn_scale = _layer_aware_attn_scale(
            self.layer_idx,
            self.N,
            seq_len,
            self.original_max_position_embeddings,
        )
        self.register_buffer(
            "cos_cached",
            (self.cos_cached * attn_scale).to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (self.sin_cached * attn_scale).to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
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
    """
    Freq-Smooth Block RoPE with position encoding only.

    Implements quantized effective position indices using a quadratic block-size
    schedule derived from normalized RoPE base frequencies. Provides C1 smoothness
    at the critical dimension boundary.

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
        theta_istar (float): Theta value at critical dimension.
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
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scaling_factor = scaling_factor
        self.layer_idx = layer_idx
        self.N = num_hidden_layers
        self.dynamic = dynamic

        # ── Base inv_freq: θ_i = base^{-2i/d}, θ_0 = 1 ─────────────── #
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # ── Critical dimension i* and anchor θ_{i*} ──────────────────── #
        # r_i = L_0 · θ_i / (2π) ≥ 1  ↔  dim completes ≥ 1 full rotation
        # i* = first index where r_i < 1
        r = original_max_position_embeddings * inv_freq / (2.0 * math.pi)
        # number of dims with r_i >= 1
        i_star = int((r >= 1.0).sum().item())
        # guard: keep i* in (0, d//2 - 1) so both zones are non-empty
        self.i_star = max(1, min(i_star, dim // 2 - 1))
        # θ at i* — used as the normalisation anchor
        self.theta_istar: float = float(inv_freq[self.i_star].item())

        if not dynamic:
            block_sizes = self._compute_block_sizes(scaling_factor, device=device)
            self.register_buffer("block_sizes", block_sizes, persistent=False)
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=device or inv_freq.device,
                dtype=torch.get_default_dtype(),
            )
        # dynamic: block_sizes recomputed each forward pass

    # ------------------------------------------------------------------ #

    def _compute_block_sizes(self, S: float, device=None) -> torch.Tensor:
        """
        Compute per-dimension block sizes using quadratic schedule.

        Uses normalized frequency to compute block sizes with C1 smoothness
        at the critical dimension boundary.

        Args:
            S (float): Extension ratio for scaling.
            device: Device for tensor creation.

        Returns:
            torch.Tensor: Float32 tensor of shape [dim//2] containing block sizes.
        """
        theta = self.inv_freq.to(device=device)  # [d//2]
        S = float(S)
        th_star = self.theta_istar  # scalar anchor

        # Normalised frequency: 1 at i=0, 0 at i=i*, <0 beyond (clamped)
        denom = max(1.0 - th_star, 1e-8)  # avoid div-by-zero
        theta_hat = torch.clamp(
            (theta - th_star) / denom,
            min=0.0,
            max=1.0,
        )  # [d//2]

        # Quadratic schedule: b_i = S − (S−1) · θ̂_i²
        b = S - (S - 1.0) * theta_hat * theta_hat  # [d//2]

        # Hard-set i ≥ i* to exactly S (avoids any residual float deviation)
        b[self.i_star :] = S

        return torch.clamp(b, min=1.0, max=S)

    # ------------------------------------------------------------------ #

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        b = self.block_sizes.to(device=device)  # [d//2]
        inv_freq = self.inv_freq.to(device=device)  # [d//2]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)  # [N]

        # t_eff[n, i] = floor(t[n] / b[i])
        t_eff = torch.floor(t[:, None] / b[None, :])  # [N, d//2]
        freqs = t_eff * inv_freq[None, :]  # [N, d//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [N, d]

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, seq_len: int = None):
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            S = max(1.0, seq_len / self.original_max_position_embeddings)
            b = self._compute_block_sizes(S, device=device)
            inv_freq = self.inv_freq.to(device=device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            t_eff = torch.floor(t[:, None] / b[None, :])
            freqs = t_eff * inv_freq[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=dtype),
                self.sin_cached[:seq_len].to(dtype=dtype),
            )


# ---------------------------------------------------------------------------- #


class LlamaFreqSmoothScaledRotaryEmbedding(LlamaFreqSmoothRotaryEmbedding):
    """
    Freq-Smooth Block RoPE with attention temperature scaling.

    Inherits position encoding from LlamaFreqSmoothRotaryEmbedding and applies
    a layer-dependent attention temperature scalar with inverted-U profile.

    Attributes:
        Inherits all attributes from LlamaFreqSmoothRotaryEmbedding.
    """

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        super()._set_cos_sin_cache(seq_len, device, dtype)
        attn_scale = _layer_aware_attn_scale(
            self.layer_idx,
            self.N,
            seq_len,
            self.original_max_position_embeddings,
        )
        self.register_buffer(
            "cos_cached",
            (self.cos_cached * attn_scale).to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (self.sin_cached * attn_scale).to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
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
    """
    Freq-Reciprocal Block RoPE — position encoding only (no attention scaling).

    Core idea
    ---------
    Like Block-Layered RoPE, each RoPE dimension i uses a quantised effective
    position index:

        t_eff(i) = floor(t / b_i)

    The block-size schedule b_i is defined so that its rate of change db_i/di
    and the rate of decay |dθ_i/di| of the RoPE base frequency satisfy a
    **constant-product (reciprocal) relationship** for all i < i*:

        db_i/di · |dθ_i/di| = const

    Because |dθ_i/di| ∝ θ_i (decelerating), the reciprocal constraint forces
    db_i/di ∝ 1/θ_i (accelerating) — matching the BlockLayered design
    philosophy while being grounded in the frequency structure of the model.

    Formula
    -------
    Let i* be the critical dimension (first index where r_i = L_0·θ_i/(2π) < 1).
    Let K = (S−1) / (1/θ_{i*} − 1).  Then:

        b_i = 1 + K · (1/θ_i − 1)    for i < i*
        b_i = S                        for i ≥ i*

    Equivalently, using θ_i = base^{−2i/d}:

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
        Static extension ratio S > 1.0.  Ignored in dynamic mode.
    dynamic : bool
        False (default) — static mode: b_i and cos/sin pre-cached at init.
        True            — dynamic mode: S = max(1, seq_len / L_0) at runtime.
    layer_idx : int
        0-based index of this attention layer.  Stored for use by
        ``LlamaFreqReciprocalScaledRotaryEmbedding``; not used here.
    num_hidden_layers : int
        Total transformer layers.  Stored for the scaled subclass.
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
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scaling_factor = scaling_factor
        self.layer_idx = layer_idx
        self.N = num_hidden_layers
        self.dynamic = dynamic

        # ── Base inv_freq: θ_i = base^{-2i/d}, θ_0 = 1 ─────────────── #
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # ── Critical dimension i* and anchor 1/θ_{i*} ───────────────── #
        # r_i = L_0 · θ_i / (2π);  i* = first index where r_i < 1
        r = original_max_position_embeddings * inv_freq / (2.0 * math.pi)
        i_star = int((r >= 1.0).sum().item())
        self.i_star = max(1, min(i_star, dim // 2 - 1))

        # 1/θ_{i*} = base^{2i*/d}  — the normalisation denominator anchor
        # Store as Python float to avoid device issues in dynamic mode.
        self.inv_theta_istar: float = float((base ** (2.0 * self.i_star / dim)))

        if not dynamic:
            block_sizes = self._compute_block_sizes(scaling_factor, device=device)
            self.register_buffer("block_sizes", block_sizes, persistent=False)
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=device or inv_freq.device,
                dtype=torch.get_default_dtype(),
            )

    # ------------------------------------------------------------------ #

    def _compute_block_sizes(self, S: float, device=None) -> torch.Tensor:
        """
        Compute per-dimension block sizes using reciprocal frequency schedule.

        Block sizes follow a linear relationship with reciprocal frequency,
        ensuring constant product between block size rate and frequency decay rate.

        Args:
            S (float): Extension ratio for scaling.
            device: Device for tensor creation.

        Returns:
            torch.Tensor: Float32 tensor of shape [dim//2] containing block sizes.
        """
        S = float(S)

        # 1/θ_i = base^{2i/d}  — reciprocal of inv_freq
        # inv_freq holds θ_i, so 1/inv_freq = 1/θ_i
        inv_theta = 1.0 / self.inv_freq.to(device=device)  # [d//2], base^{2i/d}

        # Normalisation constant K
        denom = self.inv_theta_istar - 1.0  # base^{2i*/d} - 1
        if abs(denom) < 1e-8:
            # degenerate guard (only if i*=0, which is excluded by clamping)
            return torch.ones(self.dim // 2, device=device, dtype=torch.float32) * S
        K = (S - 1.0) / denom

        # b_i = 1 + K · (1/θ_i − 1)
        b = 1.0 + K * (inv_theta - 1.0)  # [d//2]

        # Hard-set i ≥ i* to exactly S (avoids any floating-point residual)
        b[self.i_star :] = S

        return torch.clamp(b, min=1.0, max=S)

    # ------------------------------------------------------------------ #

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        b = self.block_sizes.to(device=device)  # [d//2]
        inv_freq = self.inv_freq.to(device=device)  # [d//2]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        # t_eff[n, i] = floor(t[n] / b[i])
        t_eff = torch.floor(t[:, None] / b[None, :])  # [N, d//2]
        freqs = t_eff * inv_freq[None, :]  # [N, d//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [N, d]

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, seq_len: int = None):
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            S = max(1.0, seq_len / self.original_max_position_embeddings)
            b = self._compute_block_sizes(S, device=device)
            inv_freq = self.inv_freq.to(device=device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            t_eff = torch.floor(t[:, None] / b[None, :])
            freqs = t_eff * inv_freq[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=dtype),
                self.sin_cached[:seq_len].to(dtype=dtype),
            )


# ---------------------------------------------------------------------------- #


class LlamaFreqReciprocalScaledRotaryEmbedding(LlamaFreqReciprocalRotaryEmbedding):
    """
    Freq-Reciprocal Block RoPE with power-law attention temperature scaling.

    Inherits position encoding from LlamaFreqReciprocalRotaryEmbedding and
    applies a power-law attention temperature scaling with layer-dependent factors.

    Attributes:
        alpha (float): Exponent for scaling factor S.
        beta (float): Layer-dependent scaling coefficient.
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
        """
        Compute power-law attention temperature scaling.

        Combines position and depth factors for attention temperature adjustment.

        Args:
            seq_len (int): Sequence length for scaling computation.
            device: Device for tensor creation.

        Returns:
            torch.Tensor: Tensor of shape (seq_len, 1) for broadcasting with cos/sin caches.
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
        super()._set_cos_sin_cache(seq_len, device, dtype)
        attn_scale = self._compute_attn_scale(seq_len, device)
        self.register_buffer(
            "cos_cached",
            (self.cos_cached * attn_scale).to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (self.sin_cached * attn_scale).to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if self.dynamic:
            if seq_len is None:
                seq_len = x.shape[2]
            cos, sin = super().forward(x, seq_len)
            S = max(1.0, seq_len / self.original_max_position_embeddings)
            attn_scale = self._compute_attn_scale(seq_len, x.device)
            return (cos * attn_scale).to(x.dtype), (sin * attn_scale).to(x.dtype)
        return super().forward(x, seq_len)


class LlamaFreqReciprocalScaledNoLayerRotaryEmbedding(
    LlamaFreqReciprocalRotaryEmbedding
):
    """
    Freq-Reciprocal Block RoPE with power-law attention temperature scaling (no layer index).

    Inherits position encoding from LlamaFreqReciprocalRotaryEmbedding and
    applies a power-law attention temperature scaling without layer-dependent factors.

    Attributes:
        alpha (float): Exponent for scaling factor S.
        beta (float): Layer-dependent scaling coefficient (unused in this variant).
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
        """
        Compute power-law attention temperature scaling without layer index.

        Args:
            seq_len (int): Sequence length for scaling computation.
            device: Device for tensor creation.

        Returns:
            torch.Tensor: Tensor of shape (seq_len, 1) for broadcasting with cos/sin caches.
        """
        t = torch.maximum(
            torch.tensor(1.0, device=device),
            torch.arange(seq_len, device=device, dtype=torch.float32)
            / self.original_max_position_embeddings,
        )

        # depth_factor = math.exp(self.layer_idx / self.N) / math.e

        S_t = 1.0 + 0.1 * t.log()

        return S_t.unsqueeze(-1)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        super()._set_cos_sin_cache(seq_len, device, dtype)
        attn_scale = self._compute_attn_scale(seq_len, device)
        self.register_buffer(
            "cos_cached",
            (self.cos_cached * attn_scale).to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (self.sin_cached * attn_scale).to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if self.dynamic:
            if seq_len is None:
                seq_len = x.shape[2]
            cos, sin = super().forward(x, seq_len)
            S = max(1.0, seq_len / self.original_max_position_embeddings)
            attn_scale = self._compute_attn_scale(seq_len, x.device)
            return (cos * attn_scale).to(x.dtype), (sin * attn_scale).to(x.dtype)
        return super().forward(x, seq_len)


class LlamaFreqReciprocalScaledAdaptiveRotaryEmbedding(
    LlamaFreqReciprocalRotaryEmbedding
):
    """
    Freq-Reciprocal Block RoPE with adaptive attention temperature scaling.

    Inherits position encoding from LlamaFreqReciprocalRotaryEmbedding and
    applies an adaptive attention temperature scaling that considers both
    position and dimension compression factors.

    Attributes:
        alpha (float): Exponent for scaling factor S.
        beta (float): Layer-dependent scaling coefficient.
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
        self.attn_scale_coef = attn_scale_coef

    def _compute_attn_scale(self, t_eff: torch.Tensor, device):
        """
        Compute adaptive attention temperature scaling.

        Considers both position and dimension compression factors for adaptive
        temperature adjustment based on sequence length and block sizes.

        Args:
            t_eff (torch.Tensor): Position effective length, shape (seq_len, d//2).
            device: Device for tensor creation.

        Returns:
            torch.Tensor: Tensor of shape (seq_len, d//2) for broadcasting with cos/sin caches.
        """
        """
        t = torch.maximum(
            torch.tensor(1.0, device=device),
            torch.arange(seq_len, device=device, dtype=torch.float32)
            / self.original_max_position_embeddings,
        )  # [seq_len]

        S_t = 1.0 + 0.15 * t.log()[:, None] # / block_sizes[None, :]

        t = torch.arange(seq_len, device=device, dtype=torch.float32) + 1.0
        scale = 1.0 + 0.15 * torch.exp(
            -1.0 * t / max(seq_len, self.original_max_position_embeddings)
        ) * math.log(max(1.0, seq_len / self.original_max_position_embeddings))
        S_t = torch.clamp(scale, min=1.0).unsqueeze(-1)
        """
        # s = torch.clamp(t_eff * self._inv_original_max_pos, min=1.0)
        # t_base = self.attn_scale_base + self.attn_scale_coef * torch.log(s)
        # return t_base

        t_clipped = torch.clamp(t_eff, min=1.0)
        log_t = torch.log(t_clipped)
        log_L0 = math.log(self.original_max_position_embeddings)
        normalized_log = (log_t - log_L0) / log_L0
        clipped_log = torch.clamp(normalized_log, min=0.0)
        mscale = 1.0 + self.attn_scale_coef * clipped_log

        return mscale

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        b = self.block_sizes.to(device=device)
        inv_freq = self.inv_freq.to(device=device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        t_eff = torch.floor(t[:, None] / b[None, :])
        freqs = t_eff * inv_freq[None, :]
        attn_scale = self._compute_attn_scale(t_eff, device)

        cos = (freqs.cos() * attn_scale).repeat(1, 2)
        sin = (freqs.sin() * attn_scale).repeat(1, 2)

        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            S = max(1.0, seq_len / self.original_max_position_embeddings)
            b = self._compute_block_sizes(S, device=device)
            inv_freq = self.inv_freq.to(device=device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)

            t_eff = torch.floor(t[:, None] / b[None, :])
            freqs = t_eff * inv_freq[None, :]
            attn_scale = self._compute_attn_scale(t_eff, device)

            cos = (freqs.cos() * attn_scale).repeat(1, 2)
            sin = (freqs.sin() * attn_scale).repeat(1, 2)
            return cos.to(dtype), sin.to(dtype)
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
    """
    Dual RoPE Embedding.

    A novel dual-position encoding approach that splits position indices into two parts
    based on the critical dimension i_star, while keeping inv_freq complete:

    - inv_freq: complete, size = dim // 2 (e.g., [f_0, f_1, f_2, ..., f_{dim//2-1}])
    - First part dimensions (i < i_star): position index = t % S (local, cyclic)
    - Second part dimensions (i >= i_star): position index = t // S (global, monotonic)

    The critical dimension i_star is computed as the number of dimensions that complete
    at least one full rotation within the original context window:
        r_i = L_0 * θ_i / (2π),  i_star = first index where r_i < 1

    This design allows:
    1. High-frequency dimensions (i < i_star): capture local position via modulo
    2. Low-frequency dimensions (i >= i_star): capture global position via integer division

    Attributes:
        dim (int): Dimension of the embedding (head dimension).
        max_position_embeddings (int): Maximum sequence length for caching.
        base (int): Base frequency for computing inverse frequencies.
        scaling_factor (float): Scaling factor for position interpolation.
        original_max_position_embeddings (int): Original context window size.
        dynamic (bool): Whether to dynamically recompute frequencies.
        i_star (int): Critical dimension index for splitting position indices.
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

        if not dynamic:
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=self.inv_freq.device,
                dtype=torch.get_default_dtype(),
            )

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len

        S = self.S
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        pos_1 = t % S
        pos_2 = t // S

        self.register_buffer("pos_1_cached", pos_1, persistent=False)
        self.register_buffer("pos_2_cached", pos_2, persistent=False)

        inv_freq_1 = self.inv_freq_1.to(device=device)
        inv_freq_2 = self.inv_freq_2.to(device=device)

        freqs_1 = pos_1[:, None] * inv_freq_1[None, :]
        freqs_2 = pos_2[:, None] * inv_freq_2[None, :]

        freqs = torch.cat([freqs_1, freqs_2], dim=-1)

        cos = freqs.cos().repeat(1, 2)
        sin = freqs.sin().repeat(1, 2)

        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            scaling_factor = max(1.0, seq_len / self.original_max_position_embeddings)
            S = max(scaling_factor, self.original_max_position_embeddings)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)

            pos_1 = t % S
            pos_2 = t // S

            inv_freq_1 = self.inv_freq_1.to(device=device)
            inv_freq_2 = self.inv_freq_2.to(device=device)

            freqs_1 = pos_1[:, None] * inv_freq_1[None, :]
            freqs_2 = pos_2[:, None] * inv_freq_2[None, :]

            freqs = torch.cat([freqs_1, freqs_2], dim=-1)

            cos = freqs.cos().repeat(1, 2)
            sin = freqs.sin().repeat(1, 2)

            return cos.to(dtype), sin.to(dtype)
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            return (
                self.cos_cached[:seq_len].to(dtype),
                self.sin_cached[:seq_len].to(dtype),
            )


class LlamaDualRoPEScaledEmbedding(LlamaDualRoPEEmbedding):
    """
    Dual RoPE Embedding with attention temperature scaling.

    Inherits the dual-position encoding from LlamaDualRoPEEmbedding and
    adds adaptive attention temperature scaling for better long-context handling.

    The attention scaling uses the formula: max(1.0, log_{L0}(t))
    where L0 is the original_max_position_embeddings and t is the position index.
    This is the entropy-based scaling formula that does not distinguish between dimensions.

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
        """
        Compute attention temperature scaling using entropy-based formula.

        Uses the formula: mscale = max(1.0, log_{L0}(t))
        where log_{L0}(t) = ln(t) / ln(L0).

        This ensures temperature = 1.0 when t <= L0, and scales logarithmically
        when t > L0. The scaling is applied uniformly across all dimensions.

        Args:
            seq_len (int): Sequence length.
            device: Device for tensor creation.

        Returns:
            torch.Tensor: Temperature scaling factor of shape [seq_len, 1].
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
        super()._set_cos_sin_cache(seq_len, device, dtype)

        attn_scale = self._compute_attn_scale(seq_len, device)

        self.register_buffer(
            "cos_cached",
            (self.cos_cached * attn_scale).to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (self.sin_cached * attn_scale).to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if self.dynamic:
            if seq_len is None:
                seq_len = x.shape[2]

            scaling_factor = max(1.0, seq_len / self.original_max_position_embeddings)
            S = max(scaling_factor, self.original_max_position_embeddings)
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            pos_1 = t % S
            pos_2 = t // S

            inv_freq_1 = self.inv_freq_1.to(device=x.device)
            inv_freq_2 = self.inv_freq_2.to(device=x.device)

            freqs_1 = pos_1[:, None] * inv_freq_1[None, :]
            freqs_2 = pos_2[:, None] * inv_freq_2[None, :]

            freqs = torch.cat([freqs_1, freqs_2], dim=-1)

            cos = freqs.cos().repeat(1, 2)
            sin = freqs.sin().repeat(1, 2)

            attn_scale = self._compute_attn_scale(seq_len, x.device)

            return (cos * attn_scale).to(x.dtype), (sin * attn_scale).to(x.dtype)
        return super().forward(x, seq_len)


# ================================================================================== #
#  Inverse Dual RoPE family                                                           #
#                                                                                    #
#  LlamaInverseDualRoPEEmbedding       (Inverse-Dual-RoPE, position only)             #
#  LlamaInverseDualRoPEScaledEmbedding (Inverse-Dual-RoPE + attention temperature)    #
# ================================================================================== #


class LlamaInverseDualRoPEEmbedding(nn.Module):
    """
    Inverse Dual RoPE Embedding.

    A novel inverse dual-position encoding approach that splits position indices into
    two parts based on the critical dimension i_star, while keeping inv_freq complete.
    This is the INVERSE of LlamaDualRoPEEmbedding — the high/low frequency operations
    are swapped:

    - inv_freq: complete, size = dim // 2 (e.g., [f_0, f_1, f_2, ..., f_{dim//2-1}])
    - High-frequency dimensions (i < i_star): position index = t // L_0 (global, monotonic)
    - Low-frequency dimensions (i >= i_star): position index = t % L_0 (local, cyclic)

    The critical dimension i_star is computed as the number of dimensions that complete
    at least one full rotation within the original context window:
        r_i = L_0 * θ_i / (2π),  i_star = first index where r_i < 1

    Key difference from Dual-RoPE: denominator uses L_0 (original context length),
    not S (scaled context length).

    Attributes:
        dim (int): Dimension of the embedding (head dimension).
        max_position_embeddings (int): Maximum sequence length for caching.
        base (int): Base frequency for computing inverse frequencies.
        scaling_factor (float): Scaling factor for position interpolation.
        original_max_position_embeddings (int): Original context window size.
        dynamic (bool): Whether to dynamically recompute frequencies.
        i_star (int): Critical dimension index for splitting position indices.
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
        self.max_seq_len_cached = seq_len

        L_0 = self.original_max_position_embeddings
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        pos_1 = t // L_0
        pos_2 = t % L_0

        self.register_buffer("pos_1_cached", pos_1, persistent=False)
        self.register_buffer("pos_2_cached", pos_2, persistent=False)

        inv_freq_1 = self.inv_freq_1.to(device=device)
        inv_freq_2 = self.inv_freq_2.to(device=device)

        freqs_1 = pos_1[:, None] * inv_freq_1[None, :]
        freqs_2 = pos_2[:, None] * inv_freq_2[None, :]

        freqs = torch.cat([freqs_1, freqs_2], dim=-1)

        cos = freqs.cos().repeat(1, 2)
        sin = freqs.sin().repeat(1, 2)

        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
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
    """
    Inverse Dual RoPE Embedding with attention temperature scaling.

    Inherits the inverse dual-position encoding from LlamaInverseDualRoPEEmbedding and
    adds adaptive attention temperature scaling for better long-context handling.

    The attention scaling uses the formula: mscale = max(1.0, ln(t) / ln(L_0))
    where L_0 is the original_max_position_embeddings and t is the position index.

    Attributes:
        Inherits all attributes from LlamaInverseDualRoPEEmbedding.
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
        """
        Compute attention temperature scaling using entropy-based formula.

        Uses the formula: mscale = max(1.0, ln(t) / ln(L_0))
        where L_0 is the original_max_position_embeddings.

        This ensures temperature = 1.0 when t <= L_0, and scales logarithmically
        when t > L_0. The scaling is applied uniformly across all dimensions.

        Args:
            seq_len (int): Sequence length.
            device: Device for tensor creation.

        Returns:
            torch.Tensor: Temperature scaling factor of shape [seq_len, 1].
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
        super()._set_cos_sin_cache(seq_len, device, dtype)

        attn_scale = self._compute_attn_scale(seq_len, device)

        self.register_buffer(
            "cos_cached",
            (self.cos_cached * attn_scale).to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (self.sin_cached * attn_scale).to(dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if seq_len is None:
            seq_len = x.shape[2]

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )
