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
    Position Interpolation (PI) — linear position scaling.

    Parameters
    ----------
    scaling_factor : float
        Extension ratio s.  In static mode the positions are divided by s so
        that the extended context fits into the model's original angular range.
        In dynamic mode this value is **ignored** at runtime; s is derived
        from the actual sequence length.
    original_max_position_embeddings : int
        Model's original context length L.
    dynamic : bool
        False → static pre-cached mode.
        True  → recompute on every forward pass with s = seq_len / original_L.
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
    NTK-aware RoPE scaling.

    Static mode  : modifies the RoPE base as ``base' = base * s^(d/(d-2))``
                   once at construction time and pre-caches frequencies.
    Dynamic mode : recomputes ``base'`` on every forward pass using the
                   actual scaling factor ``s = max(1, seq_len / original_L)``.
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
    NTK-by-parts RoPE scaling.

    Dimensions are split into three groups based on how many full rotations
    each dimension performs within the original training context ``original_L``:

    * High-frequency  (many rotations, ``r_d > beta``)  → kept unchanged (w=1)
    * Low-frequency   (few rotations,  ``r_d < alpha``) → linearly interpolated
      by dividing by s (w=0)
    * Transition region → smooth linear blend between the two extremes

    Parameters
    ----------
    alpha, beta : float
        Boundary parameters for the piecewise blending.  The original YaRN
        paper uses alpha=1 (≈ 1 full rotation) and beta=32 (≈ 32 full
        rotations).
    dynamic : bool
        False → static pre-cached mode with fixed ``scaling_factor``.
        True  → dynamic mode; ``scaling_factor`` recomputed each forward pass
                as ``s = max(1, seq_len / original_L)``.  In dynamic mode, the
                blending weights ``w_ext`` AND ``inv_freq`` are both fully
                recomputed on every call (Method A).
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
    YaRN = NTK-by-parts + attention temperature scaling.

    The attention temperature scale is ``t = 1 + 0.1 * ln(s)`` and is
    multiplied into the cached (or on-the-fly) cos/sin values so that
    downstream attention dot-products are automatically re-scaled.

    In static mode the scale is a fixed scalar baked into the cache.
    In dynamic mode the scale is recomputed from the runtime ``s``.
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
    Layer-dependent attention amplitude scalar (inverted-U profile).

    Middle layers receive a weaker correction (u_norm → 1, layer_alpha → 0);
    first and last layers receive a stronger correction (u_norm → 0,
    layer_alpha → 0.1).

    Returns a Python float suitable for direct multiplication with cos/sin
    tensors or cached buffers.
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
    Layer-aware My RoPE — position encoding only (no attention scaling).

    Implements NTK-by-parts frequency blending with layer-adaptive α/β boundaries
    (inverted-U profile across layers).  Attention temperature correction is
    intentionally absent; use ``LlamaMyScaledRotaryEmbedding`` for the full
    YaRN-style variant that also applies a layer-aware attention scalar.

    Parameters
    ----------
    scaling_factor : float
        Static extension ratio s > 1.0.  Used only when ``dynamic=False``.
    dynamic : bool
        False (default) — static mode: cos/sin pre-cached at init.
        True            — dynamic mode: S = max(seq_len, L) / L at runtime.

    Internals
    ---------
    * Inverted-U α/β: middle layers use wider transition bands; first/last
      layers use tighter bands (closer to standard NTK-by-parts).
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
        """Pre-compute the per-dimension blending mask (layer-adaptive)."""
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
    Layer-aware My RoPE + attention temperature scaling.

    Inherits all position-encoding logic from ``LlamaMyRotaryEmbedding`` and
    multiplies the resulting cos/sin values by a layer-dependent global scalar:

        scale = 1 + layer_alpha * log(max(1, seq_len / L_0))
        layer_alpha = 0.1 * (1 - u_norm),  u_norm = 1 - layer_norm^2

    This mirrors the relationship between ``LlamaNTKByPartsScaledRotaryEmbedding``
    (base) and ``LlamaYarnScaledRotaryEmbedding`` (scaled).

    In static mode the scalar is baked into the cos/sin cache at construction.
    In dynamic mode it is recomputed on every forward pass.
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
    Multi-scale My RoPE — position encoding only (no attention scaling).

    Splits the head dimension into three sub-spaces tuned for local (40 %),
    paragraph (30 %), and document (30 %) scales, each with its own NTK-by-parts
    parameters and base frequency.  Attention temperature correction is
    intentionally absent; use ``LlamaMyScaledRotaryEmbedding2`` for the
    YaRN-style variant with layer-aware attention scaling.

    Parameters
    ----------
    scaling_factor : float
        Static extension ratio s > 1.0.  Used only when ``dynamic=False``.
        Each sub-space blends its inv_freq using the shared ``scaling_factor``
        as the denominator (same NTK-by-parts formula as the other classes).
    dynamic : bool
        False (default) — static mode: each sub-space is scaled by the fixed
                          ``scaling_factor``; combined cos/sin are pre-cached.
        True            — dynamic mode: each sub-space derives its own scaling
                          factor from ``seq_len`` at runtime (original behaviour).
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
        """Compute blended inv_freq for one sub-space using the fixed scaling_factor."""
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
        """Compute blended inv_freq for one sub-space using the runtime seq_len."""
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
    Multi-scale My RoPE 2 + attention temperature scaling.

    Inherits all position-encoding logic from ``LlamaMyRotaryEmbedding2`` and
    multiplies the resulting cos/sin values by the same layer-dependent global
    scalar used by ``LlamaMyScaledRotaryEmbedding``:

        scale = 1 + layer_alpha * log(max(1, seq_len / L_0))

    In static mode the scalar is baked into the cos/sin cache at construction.
    In dynamic mode it is recomputed on every forward pass.
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
    Block-Layered RoPE — position encoding only (no attention scaling).

    Core idea
    ---------
    For each RoPE dimension i, an effective position index is computed as:

        t_eff(i) = floor(t / b_i)

    where b_i is a per-dimension block size that grows exponentially with i:

        b_i = S^(i / i*)        for i < i*    (high-frequency region)
        b_i = S                  for i >= i*   (low-frequency region)

    i* is the *critical dimension*: the first dimension whose wavelength
    lambda_i = 2π / θ_i exceeds the original context length L_0, i.e. the
    first dimension that never completes a full rotation within L_0.

        i* = min{ i : L_0 / λ_i  < 1 }

    Physical interpretation
    -----------------------
    * i < i*  (high-frequency, r_i = L_0/λ_i ≥ 1):
        Block size grows from b_0=1 (full resolution, standard RoPE) toward
        b_{i*-1} ≈ S (mild compression).  These dimensions resolve *local /
        within-block* token order.

    * i >= i*  (low-frequency, r_i < 1):
        Block size is fixed at S.  Effective positions are compressed to
        [0, ceil(N/S)], which stays within the original angular range
        [0, L_0 * θ_i] — preventing the angular value OOD that would
        otherwise occur when t > L_0.  These dimensions resolve *long-range /
        cross-block* structure.

    Block-size schedule
    -------------------
    Exponential growth (b_i = S^(i/i*)) is chosen because RoPE frequencies
    θ_i themselves decay geometrically with i.  An exponential b_i schedule
    keeps the effective angular range N*θ_i/b_i approximately uniform across
    all dimensions, matching the geometric structure of the frequency grid.

    Degradation
    -----------
    When seq_len <= original_L:  S = 1  →  b_i = 1 for all i  →  standard RoPE.

    Attention scaling
    -----------------
    Layer-dependent inverted-U global scalar, identical to LlamaMyRotaryEmbedding:

        layer_alpha = 0.1 * (1 - u_norm),   u_norm = 1 - layer_norm^2
        scale = 1 + layer_alpha * log(max(1, seq_len / L_0))

    Parameters
    ----------
    scaling_factor : float
        Static extension ratio S > 1.0.  Ignored in dynamic mode.
    dynamic : bool
        False (default) — static mode: b_i and cos/sin pre-cached at init.
        True            — dynamic mode: S = max(1, seq_len / L_0) at runtime.
    layer_idx : int
        Index of this attention layer (0-based).  Stored for use by
        ``LlamaBlockLayeredScaledRotaryEmbedding``; not used in this base class.
    num_hidden_layers : int
        Total number of transformer layers.  Stored for use by the scaled subclass.
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
        Per-dimension block sizes for extension ratio S.

        b_i = clamp(S^(i / i*), 1, S)   for i < i*   (exponential, 1 → S)
        b_i = S                           for i >= i*  (full compression)

        Returns float32 tensor of shape [dim//2].
        S = 1.0  →  all b_i = 1.0  (standard RoPE).
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
    Block-Layered RoPE + attention temperature scaling.

    Inherits all position-encoding logic from ``LlamaBlockLayeredRotaryEmbedding``
    and multiplies the resulting cos/sin values by the same layer-dependent global
    scalar used throughout the My RoPE family:

        scale = 1 + layer_alpha * log(max(1, seq_len / L_0))
        layer_alpha = 0.1 * (1 - u_norm),  u_norm = 1 - layer_norm^2

    In static mode the scalar is baked into the cos/sin cache at construction.
    In dynamic mode it is recomputed on every forward pass.
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
    Freq-Smooth Block RoPE — position encoding only (no attention scaling).

    Core idea
    ---------
    Like Block-Layered RoPE, each RoPE dimension i uses a quantised effective
    position index:

        t_eff(i) = floor(t / b_i)

    The block-size schedule b_i is derived from the RoPE base frequency θ_i
    via a **normalised quadratic** that satisfies three design requirements
    simultaneously:

    Requirements
    ------------
    (R1) Value range  : b_i ∈ [1, S] for all i, with b_0 = 1 and b_i = S
         for all i ≥ i*.
    (R2) Strong freq-correlation : b_i is a monotone function of θ_i, with
         rate of change proportional to θ_i (decelerating, matching θ_i decay).
    (R3) C¹ smooth at i*  : db_i/di → 0 as i → i*⁻, matching the zero
         derivative of the constant S plateau on the right.

    Formula
    -------
    Let i* be the critical dimension — the first index where r_i < 1, i.e.
    the dimension whose wavelength first exceeds the original context L_0:

        θ_i   = base^{−2i/d}              (RoPE base frequency; θ_0 = 1)
        r_i   = L_0 · θ_i / (2π)          (rotations within L_0)
        i*    = min{ i : r_i < 1 }

    Normalised frequency (maps θ_0 → 1, θ_{i*} → 0):

        θ̂_i  = (θ_i − θ_{i*}) / (1 − θ_{i*})

    Block-size schedule:

        b_i = S − (S − 1) · θ̂_i²    for i < i*
        b_i = S                        for i ≥ i*

    Why the quadratic?
    ------------------
    Because θ̂_{i*} = 0, the derivative dθ̂²/di|_{i*} = 2θ̂_{i*}·dθ̂/di = 0
    regardless of dθ̂/di, so b_i meets S with zero slope — C¹ continuity is
    a purely algebraic consequence, not a heuristic patch.

    Behaviour by zone
    -----------------
    i = 0      : b_0 = S − (S−1)·1 = 1  (full resolution)
    0 < i < i* : b_i grows with decelerating speed, tracking θ̂_i² ∝ θ_i²
    i = i*−1   : db_{i*-1}/di ≈ 0  (smooth approach to S)
    i ≥ i*     : b_i = S exactly  (full compression, no OOD angles)

    Degradation
    -----------
    S = 1  →  b_i = 1 for all i  (standard RoPE, no quantisation).
    seq_len ≤ original_L (dynamic mode)  →  S = 1  →  same.

    Parameters
    ----------
    scaling_factor : float
        Static extension ratio S > 1.0.  Ignored in dynamic mode.
    dynamic : bool
        False (default) — static mode: b_i and cos/sin pre-cached at init.
        True            — dynamic mode: S = max(1, seq_len / L_0) at runtime.
    layer_idx : int
        0-based index of this attention layer.  Stored for use by
        ``LlamaFreqSmoothScaledRotaryEmbedding``; not used in this base class.
    num_hidden_layers : int
        Total transformer layers.  Stored for use by the scaled subclass.
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
        Per-dimension block sizes for extension ratio S.

        Formula
        -------
        θ̂_i = clamp( (θ_i − θ_{i*}) / (1 − θ_{i*}), 0, 1 )
        b_i  = S − (S−1) · θ̂_i²          (i < i*:  quadratic, 1 → S)
        b_i  = S                            (i ≥ i*:  constant S)
        b_i  = clamp(b_i, 1, S)            (numerical safety)

        The clamp on θ̂_i is redundant by construction (θ_i is monotone and
        θ_{i*} is its value at the boundary) but guards against floating-point
        edge cases.

        S = 1.0  →  b_i = 1 for all i  (standard RoPE).
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
    Freq-Smooth Block RoPE + attention temperature scaling.

    Inherits all position-encoding logic from
    ``LlamaFreqSmoothRotaryEmbedding`` and multiplies the resulting cos/sin
    values by the layer-dependent global scalar shared across the My RoPE /
    Block-Layered family:

        scale       = 1 + layer_alpha · log(max(1, seq_len / L_0))
        layer_alpha = 0.1 · (1 − u_norm)
        u_norm      = 1 − layer_norm²
        layer_norm  = 2 · layer_idx / (N − 1) − 1

    The inverted-U profile gives stronger correction to the first and last
    transformer layers (low attention entropy) and weaker correction to the
    middle layers (high attention entropy).

    In static mode the scalar is baked into the cos/sin cache at construction.
    In dynamic mode it is recomputed on every forward pass.
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
        Per-dimension block sizes for extension ratio S.

        Formula
        -------
        inv_θ_i  = 1 / θ_i = base^{2i/d}           (reciprocal frequency)
        K        = (S − 1) / (inv_θ_{i*} − 1)       (normalisation constant)
        b_i      = 1 + K · (inv_θ_i − 1)            (i < i*: linear in 1/θ_i)
        b_i      = S                                  (i ≥ i*: exact, hard-set)
        b_i      = clamp(b_i, 1, S)                  (numerical safety)

        Rate properties
        ---------------
        db_i/di = K · (2·ln·base/d) · inv_θ_i  →  accelerating (∝ 1/θ_i)
        db_i/di · |dθ_i/di| = K · (2·ln·base/d)²  →  constant for all i < i*

        S = 1.0  →  K = 0  →  b_i = 1 for all i  (standard RoPE).
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
    Freq-Reciprocal Block RoPE + power-law attention temperature scaling.

    Inherits all position-encoding logic from
    ``LlamaFreqReciprocalRotaryEmbedding`` and multiplies the resulting cos/sin
    values by a power-law attention temperature:

        scale = √(head_dim) × S^α × (1 + β × (layer_idx / N))

    where:
        - head_dim: dimension of each attention head
        - S: sequence length extension ratio = max(1, seq_len / L_0)
        - α (alpha): exponent for scaling factor S
        - β (beta): layer-dependent scaling coefficient
        - layer_idx: 0-based index of this attention layer
        - N: total number of transformer layers

    In static mode the scalar is baked into the cos/sin cache at construction.
    In dynamic mode it is recomputed on every forward pass.
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

        The scaling factor combines position and depth factors:
            S_t = 1 + 0.1 * log(t) * (1 + depth_factor)

        where:
            - t: normalized position factor, max(1, pos / L_0)
            - depth_factor: layer-aware factor, exp(layer_idx / N) / e

        Returns:
            Tensor of shape (seq_len, 1) for broadcasting with cos/sin caches.
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
    Freq-Reciprocal Block RoPE + power-law attention temperature scaling, no layer index.

    Inherits all position-encoding logic from
    ``LlamaFreqReciprocalRotaryEmbedding`` and multiplies the resulting cos/sin
    values by a power-law attention temperature:

        scale = √(head_dim) × S^α × (1 + β × (layer_idx / N))

    where:
        - head_dim: dimension of each attention head
        - S: sequence length extension ratio = max(1, seq_len / L_0)
        - α (alpha): exponent for scaling factor S
        - β (beta): layer-dependent scaling coefficient
        - layer_idx: 0-based index of this attention layer
        - N: total number of transformer layers

    In static mode the scalar is baked into the cos/sin cache at construction.
    In dynamic mode it is recomputed on every forward pass.
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

        The scaling factor combines position and depth factors:
            S_t = 1 + 0.1 * log(t)

        where:
            - t: normalized position factor, max(1, pos / L_0)
            - depth_factor: layer-aware factor, exp(layer_idx / N) / e

        Returns:
            Tensor of shape (seq_len, 1) for broadcasting with cos/sin caches.
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
