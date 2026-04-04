"""Position encoding comparison visualization for long context extension methods.

This module generates a 2x2 subplot figure comparing the cosine values of
different Rotary Position Embedding (RoPE) variants across position indices
and embedding dimensions. The four methods visualized are:

    - Standard RoPE: Baseline rotary position encoding without any scaling.
    - Linear Interpolation: Simple position scaling by a constant factor.
    - Freq-Reciprocal: Block-wise frequency scaling that applies different
      scaling factors per dimension to extend context length while preserving
      resolution for lower-frequency dimensions.
    - NTK-By-Parts: Frequency-band-dependent scaling that applies different
      strategies based on the number of rotations within the original context
      length, with smooth transitions between bands.

The visualization uses scatter plots to show cosine values for selected
dimensions, making it easy to compare how each method distributes positional
information across the extended context window.

Output:
    drawer/rope_comparison.png: 2x2 comparison figure at 300 DPI.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import matplotlib.pyplot as plt


def get_rope_curve(dim, max_pos, base=10000):
    """Compute the cosine values of standard RoPE (Rotary Position Embedding).

    Calculates the cosine component of the rotary embedding for each position
    and dimension pair using the standard RoPE formulation with fixed inverse
    frequencies derived from the base value.

    Args:
        dim (int): The dimension of the embedding space. Must be even.
        max_pos (int): The maximum position index (exclusive upper bound).
        base (float, optional): The base value for computing inverse frequencies.
            Defaults to 10000.

    Returns:
        torch.Tensor: Tensor of shape (max_pos, dim//2) containing cosine values
            for each position-dimension pair.
    """
    d_half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(max_pos, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    return freqs.cos()


def get_freq_reciprocal_curve(dim, max_pos, base=10000, orig_L0=2048, S=8.0):
    """Compute the cosine values of Freq-Reciprocal position encoding.

    This method applies a block-wise scaling factor to positions to extend
    the effective context length while maintaining resolution for lower
    dimensions. Dimensions below the threshold i_star receive linearly
    interpolated scaling, while dimensions at or above i_star receive the
    full scaling factor S.

    Args:
        dim (int): The dimension of the embedding space. Must be even.
        max_pos (int): The maximum position index (exclusive upper bound).
        base (float, optional): The base value for computing inverse frequencies.
            Defaults to 10000.
        orig_L0 (int, optional): The original training context length in tokens.
            Defaults to 2048.
        S (float, optional): The scaling factor for high-frequency dimensions.
            Must be greater than 1.0. Defaults to 8.0.

    Returns:
        torch.Tensor: Tensor of shape (max_pos, dim//2) containing cosine values
            for each position-dimension pair under Freq-Reciprocal scaling.
    """
    d_half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # Calculate i_star: the threshold dimension index separating scaled and unscaled regions
    r = orig_L0 * inv_freq / (2 * math.pi)
    i_star = int((r >= 1.0).sum().item())
    i_star = max(1, min(i_star, d_half - 1))

    # Compute block b_i: dimension-specific scaling factors
    inv_theta_istar = base ** (2.0 * i_star / dim)
    denom = inv_theta_istar - 1.0
    K = (S - 1.0) / denom if abs(denom) > 1e-8 else 0.0

    inv_theta = 1.0 / inv_freq
    b = 1.0 + K * (inv_theta - 1.0)
    b[i_star:] = S
    b = torch.clamp(b, 1.0, S)

    # Compute effective positions after applying per-dimension scaling
    pos = torch.arange(max_pos, dtype=torch.float32)
    t_eff = torch.floor(pos[:, None] / b[None, :])
    freqs = t_eff * inv_freq[None, :]
    return freqs.cos()


def get_ntk_by_parts_curve(
    dim,
    max_pos,
    base=10000,
    orig_L0=2048,
    scaling_factor=8.0,
    alpha=1.0,
    beta=32.0,
):
    """Compute the cosine values of NTK-By-Parts position encoding.

    This method applies different scaling strategies to different frequency
    bands based on the number of rotations within the original context length:

        - High frequency (r_d >= beta): No scaling applied to preserve local information.
        - Low frequency (r_d <= alpha): Linear interpolation by scaling_factor.
        - Middle frequency (alpha < r_d < beta): Smooth transition between the two regimes.

    The rotation count r_d = orig_L0 / lambda_d, where lambda_d is the wavelength.
    Higher r_d indicates more rotations within the original context, corresponding
    to higher frequency dimensions.

    Args:
        dim (int): The dimension of the embedding space. Must be even.
        max_pos (int): The maximum position index (exclusive upper bound).
        base (float, optional): The base value for computing inverse frequencies.
            Defaults to 10000.
        orig_L0 (int, optional): The original training context length in tokens.
            Defaults to 2048.
        scaling_factor (float, optional): The extension ratio S for context length.
            Defaults to 8.0.
        alpha (float, optional): Lower threshold for rotation count. Dimensions with
            r_d <= alpha receive full scaling. Defaults to 1.0.
        beta (float, optional): Upper threshold for rotation count. Dimensions with
            r_d >= beta receive no scaling. Defaults to 32.0.

    Returns:
        torch.Tensor: Tensor of shape (max_pos, dim//2) containing cosine values
            for each position-dimension pair under NTK-By-Parts scaling.
    """
    theta_d = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    lambda_d = 2 * math.pi / theta_d
    r_d = orig_L0 / lambda_d
    w_ext = torch.clamp((r_d - alpha) / (beta - alpha), 0.0, 1.0)
    inv_freq = theta_d * w_ext + (1.0 - w_ext) * theta_d / scaling_factor
    pos = torch.arange(max_pos, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    return freqs.cos()


def get_linear_curve(dim, max_pos, base=10000, scale_factor=8.0):
    """Compute the cosine values of Linear Interpolation position encoding.

    This method uniformly scales all positions by a constant factor to extend
    the context length. It is the simplest approach for length extrapolation,
    equivalent to Position Interpolation (PI).

    Args:
        dim (int): The dimension of the embedding space. Must be even.
        max_pos (int): The maximum position index (exclusive upper bound).
        base (float, optional): The base value for computing inverse frequencies.
            Defaults to 10000.
        scale_factor (float, optional): The uniform scaling factor for all positions.
            Defaults to 8.0.

    Returns:
        torch.Tensor: Tensor of shape (max_pos, dim//2) containing cosine values
            for each position-dimension pair under linear scaling.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(max_pos, dtype=torch.float32)
    scaled_pos = pos / scale_factor
    freqs = scaled_pos[:, None] * inv_freq[None, :]
    return freqs.cos()


# --------------------------
# Visualization parameters
# --------------------------
DIM = 128
BASE = 15
ORIG_L0 = 64
SCALE_FACTOR = 8.0
ALPHA = 1.0
BETA = 32.0

ROPE_POS_MAX = 63
FRQ_POS_MAX = 511
NTK_POS_MAX = 511
LINEAR_POS_MAX = 511

# Dimensions to visualize (selected for clear contrast)
VIS_DIMS = [30, 60]
COLORS = ["#d62728", "#1f77b4"]  # Red for low dimension, Blue for high dimension

# --------------------------
# Compute curves for selected dimensions
# --------------------------
rope_cos_dict = {}
frq_cos_dict = {}
ntk_cos_dict = {}
linear_cos_dict = {}

for vis_dim in VIS_DIMS:
    rope_cos_dict[vis_dim] = get_rope_curve(DIM, ROPE_POS_MAX + 1, BASE)[:, vis_dim]
    frq_cos_dict[vis_dim] = get_freq_reciprocal_curve(
        DIM,
        FRQ_POS_MAX + 1,
        BASE,
        ORIG_L0,
        SCALE_FACTOR,
    )[:, vis_dim]
    ntk_cos_dict[vis_dim] = get_ntk_by_parts_curve(
        DIM,
        NTK_POS_MAX + 1,
        BASE,
        ORIG_L0,
        SCALE_FACTOR,
        ALPHA,
        BETA,
    )[:, vis_dim]
    linear_cos_dict[vis_dim] = get_linear_curve(
        DIM,
        LINEAR_POS_MAX + 1,
        BASE,
        SCALE_FACTOR,
    )[:, vis_dim]

# --------------------------
# Generate 2x2 comparison plot
# --------------------------
fig = plt.figure(figsize=(14, 12), dpi=120)
fig.suptitle(
    "Position Encoding Comparison for Long Context Extension",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# Top-left plot: Standard RoPE (baseline, short context)
ax1 = plt.subplot(2, 2, 1)
for vis_dim, color in zip(VIS_DIMS, COLORS):
    plt.scatter(
        torch.arange(ROPE_POS_MAX + 1).numpy(),
        rope_cos_dict[vis_dim].numpy(),
        color=color,
        s=30,
        alpha=0.7,
        label=f"dim={vis_dim}",
    )
plt.title(f"Standard RoPE | Position 0~63", fontsize=14, fontweight=600)
plt.ylabel("Cosine Value", fontsize=12)
plt.xticks([0, 10, 20, 30, 40, 50, 60])
plt.grid(alpha=0.3, linestyle="--")
plt.ylim(-1.1, 1.1)

# Top-right plot: Linear Interpolation
ax2 = plt.subplot(2, 2, 2)
for vis_dim, color in zip(VIS_DIMS, COLORS):
    plt.scatter(
        torch.arange(LINEAR_POS_MAX + 1).numpy(),
        linear_cos_dict[vis_dim].numpy(),
        color=color,
        s=30,
        alpha=0.7,
        label=f"dim={vis_dim}",
    )
plt.title(
    f"Linear (scale={SCALE_FACTOR}) | Position 0~511",
    fontsize=14,
    fontweight=600,
)
plt.xticks([0, 80, 160, 240, 320, 400, 480])
plt.grid(alpha=0.3, linestyle="--")
plt.ylim(-1.1, 1.1)

# Bottom-left plot: Freq-Reciprocal
ax3 = plt.subplot(2, 2, 3)
for vis_dim, color in zip(VIS_DIMS, COLORS):
    plt.scatter(
        torch.arange(FRQ_POS_MAX + 1).numpy(),
        frq_cos_dict[vis_dim].numpy(),
        color=color,
        s=30,
        alpha=0.7,
        label=f"dim={vis_dim}",
    )
plt.title(
    f"Freq-Reciprocal (S={SCALE_FACTOR}) | Position 0~511",
    fontsize=14,
    fontweight=600,
)
plt.xlabel("Position", fontsize=12)
plt.ylabel("Cosine Value", fontsize=12)
plt.xticks([0, 80, 160, 240, 320, 400, 480])
plt.grid(alpha=0.3, linestyle="--")
plt.ylim(-1.1, 1.1)

# Bottom-right plot: NTK-By-Parts
ax4 = plt.subplot(2, 2, 4)
for vis_dim, color in zip(VIS_DIMS, COLORS):
    plt.scatter(
        torch.arange(NTK_POS_MAX + 1).numpy(),
        ntk_cos_dict[vis_dim].numpy(),
        color=color,
        s=30,
        alpha=0.7,
        label=f"dim={vis_dim}",
    )
plt.title(
    f"NTK-By-Parts (S={SCALE_FACTOR}) | Position 0~511",
    fontsize=14,
    fontweight=600,
)
plt.xlabel("Position", fontsize=12)
plt.xticks([0, 80, 160, 240, 320, 400, 480])
plt.grid(alpha=0.3, linestyle="--")
plt.ylim(-1.1, 1.1)

# Place shared legend at bottom center
handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=2,
    fontsize=12,
    frameon=True,
    fancybox=True,
    shadow=False,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.05)
plt.savefig("drawer/rope_comparison.png", dpi=300)
