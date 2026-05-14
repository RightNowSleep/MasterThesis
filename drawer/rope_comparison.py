"""Position encoding comparison visualization for long context extension methods.

This module generates a 1x3 subplot figure comparing the cosine values of
different Rotary Position Embedding (RoPE) variants across position indices
and embedding dimensions. The three methods visualized are:

    - Standard RoPE: Baseline rotary position encoding without any scaling.
    - Position Interpolation (PI): Simple position scaling by a constant factor.
    - BiSpace-RoPE: Inverse dual-position encoding that splits position indices
      into high-frequency (global monotonic) and low-frequency (cyclic) parts.

The visualization uses scatter plots to show cosine values for selected
dimensions, making it easy to compare how each method distributes positional
information across the extended context window.

Output:
    drawer/rope_comparison.png: 1x3 comparison figure at 300 DPI.
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
        torch.Tensor: Tensor of shape (max_pos) containing cosine values
            for each position index.
    """
    inv_freq = 1.0 / (base ** ((2 * dim) / 128))
    pos = torch.arange(max_pos, dtype=torch.float32)
    freqs = pos * inv_freq
    return torch.round(freqs.cos(), decimals=4)


def get_linear_curve(dim, max_pos, base=10000, scale_factor=8.0):
    """Compute the cosine values of Position Interpolation (PI) position encoding.

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
    inv_freq = 1.0 / (base ** ((2 * dim) / 128))
    pos = torch.arange(max_pos, dtype=torch.float32)
    scaled_pos = pos / scale_factor
    freqs = scaled_pos * inv_freq
    return torch.round(freqs.cos(), decimals=4)


def get_bispace_rope_curve(dim, max_pos, base=10000, orig_L0=2048):
    """Compute the cosine values of BiSpace-RoPE (Inverse Dual RoPE) position encoding.

    This method splits position indices into two parts based on the critical
    dimension i_star:
        - High-frequency dimensions (i < i_star): position index = t (global, monotonic)
        - Low-frequency dimensions (i >= i_star): position index = t % L_0 (cyclic)

    Args:
        dim (int): The dimension of the embedding space. Must be even.
        max_pos (int): The maximum position index (exclusive upper bound).
        base (float, optional): The base value for computing inverse frequencies.
            Defaults to 10000.
        orig_L0 (int, optional): The original training context length in tokens.
            Defaults to 2048.

    Returns:
        torch.Tensor: Tensor of shape (max_pos) containing cosine values
            for each position index under BiSpace-RoPE encoding.
    """
    inv_freq = 1.0 / (base ** ((2 * dim) / 128))
    pos = torch.arange(max_pos, dtype=torch.float32)
    pos = pos % orig_L0
    freqs = pos * inv_freq
    return torch.round(freqs.cos(), decimals=4)


VIS_DIMS = [50, 60]
COLORS = ["#d62728", "#1f77b4"]
MISSING_COLOR = "#2ca02c"

DIM = 128
BASE = 15
ORIG_L0 = 64
SCALE_FACTOR = 8.0

ROPE_POS_MAX = 63
PI_POS_MAX = 511
BISPACE_POS_MAX = 511

rope_cos_dict = {}
pi_cos_dict = {}
bispace_cos_dict = {}

for vis_dim in VIS_DIMS:
    rope_cos_dict[vis_dim] = get_rope_curve(
        vis_dim,
        ROPE_POS_MAX + 1,
        BASE,
    )
    pi_cos_dict[vis_dim] = get_linear_curve(
        vis_dim,
        PI_POS_MAX + 1,
        BASE,
        SCALE_FACTOR,
    )
    bispace_cos_dict[vis_dim] = get_bispace_rope_curve(
        vis_dim,
        BISPACE_POS_MAX + 1,
        BASE,
        ORIG_L0,
    )

rope_value_sets = {}
for vis_dim in VIS_DIMS:
    rope_values = rope_cos_dict[vis_dim].numpy()
    rope_value_sets[vis_dim] = set(rope_values)

pi_missing_masks = {}
bispace_missing_masks = {}
for vis_dim in VIS_DIMS:
    rope_set = rope_value_sets[vis_dim]
    pi_values = pi_cos_dict[vis_dim].numpy()
    pi_missing_masks[vis_dim] = torch.tensor([v not in rope_set for v in pi_values])
    bispace_values = bispace_cos_dict[vis_dim].numpy()
    bispace_missing_masks[vis_dim] = torch.tensor(
        [v not in rope_set for v in bispace_values]
    )

fig = plt.figure(figsize=(18, 6), dpi=120)
fig.suptitle(
    "Position Encoding Comparison for Long Context Extension",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

ax1 = plt.subplot(1, 3, 1)
for idx, vis_dim in enumerate(VIS_DIMS):
    color = COLORS[idx]
    plt.scatter(
        torch.arange(ROPE_POS_MAX + 1).numpy(),
        rope_cos_dict[vis_dim].numpy(),
        color=color,
        s=30,
        alpha=0.7,
        label=f"dim={vis_dim}",
    )
plt.title(f"Standard RoPE | Position 0~63", fontsize=14, fontweight=600)
plt.xlabel("Position", fontsize=12)
plt.ylabel("Cosine Value", fontsize=12)
plt.xticks([0, 10, 20, 30, 40, 50, 60])
plt.grid(alpha=0.3, linestyle="--")
plt.ylim(-1.1, 1.1)

ax2 = plt.subplot(1, 3, 2)
for idx, vis_dim in enumerate(VIS_DIMS):
    color = COLORS[idx]
    missing_mask = pi_missing_masks[vis_dim]
    present_mask = ~missing_mask
    present_positions = torch.arange(PI_POS_MAX + 1)[present_mask]
    present_values = pi_cos_dict[vis_dim].numpy()[present_mask]
    missing_positions = torch.arange(PI_POS_MAX + 1)[missing_mask]
    missing_values = pi_cos_dict[vis_dim].numpy()[missing_mask]
    plt.scatter(
        present_positions.numpy(),
        present_values,
        color=color,
        s=30,
        alpha=0.7,
        label=f"dim={vis_dim}",
    )
    plt.scatter(
        missing_positions.numpy(),
        missing_values,
        color=MISSING_COLOR,
        s=30,
        alpha=0.7,
        marker="x",
    )
plt.title(
    f"Position Interpolation (scale={SCALE_FACTOR}) | Position 0~511",
    fontsize=14,
    fontweight=600,
)
plt.xlabel("Position", fontsize=12)
plt.xticks([0, 80, 160, 240, 320, 400, 480])
plt.grid(alpha=0.3, linestyle="--")
plt.ylim(-1.1, 1.1)

ax3 = plt.subplot(1, 3, 3)
for idx, vis_dim in enumerate(VIS_DIMS):
    color = COLORS[idx]
    missing_mask = bispace_missing_masks[vis_dim]
    present_mask = ~missing_mask
    present_positions = torch.arange(BISPACE_POS_MAX + 1)[present_mask]
    present_values = bispace_cos_dict[vis_dim].numpy()[present_mask]
    missing_positions = torch.arange(BISPACE_POS_MAX + 1)[missing_mask]
    missing_values = bispace_cos_dict[vis_dim].numpy()[missing_mask]
    plt.scatter(
        present_positions.numpy(),
        present_values,
        color=color,
        s=30,
        alpha=0.7,
        label=f"dim={vis_dim}",
    )
    plt.scatter(
        missing_positions.numpy(),
        missing_values,
        color=MISSING_COLOR,
        s=30,
        alpha=0.7,
        marker="x",
    )
plt.title(
    f"BiSpace-RoPE (L0={ORIG_L0}) | Position 0~511",
    fontsize=14,
    fontweight=600,
)
plt.xlabel("Position", fontsize=12)
plt.xticks([0, 80, 160, 240, 320, 400, 480])
plt.grid(alpha=0.3, linestyle="--")
plt.ylim(-1.1, 1.1)

handles, labels = ax1.get_legend_handles_labels()
handles.append(
    plt.scatter(
        [], [], color=MISSING_COLOR, s=30, alpha=0.7, marker="x", label="Not in RoPE"
    )
)
labels.append("Not in RoPE")
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0),
    ncol=3,
    fontsize=12,
    frameon=True,
    fancybox=True,
    shadow=False,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig("drawer/rope_comparison.png", dpi=300)
