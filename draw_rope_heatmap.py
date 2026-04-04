"""RoPE extrapolation weight heatmap generator for NTK-by-parts method.

Computes and visualizes per-layer, per-dimension extrapolation weights for
the NTK-by-parts context extension strategy. The heatmap shows which frequency
dimensions are treated as interpolation (weight ≈ 0) versus extrapolation
(weight ≈ 1.0) across all transformer layers, using an inverted-U-shaped
normalization pattern that varies by layer depth.

The output is a high-resolution PNG image saved as ``heatmap_U2_max.png``.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# Configure Chinese font and display settings for matplotlib
plt.rcParams["font.family"] = ["SimHei", "AR PL UKai CN", "AR PL UMing CN"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def ntk_by_parts_weights_by_layer(
    dim: int = 128,
    base: int = 10000,
    L: int = 2048,
    alpha: float = 1.0,
    beta: float = 32.0,
    layer_idx: int = 0,
    N: int = 32,
) -> np.ndarray:
    """Calculate extrapolation weights for NTK-by-parts method at a given layer.

    Determines which frequency groups should be interpolated (within training
    length) versus extrapolated (beyond training length) based on their wavelength
    relative to the target sequence length *L*. The boundary between interpolation
    and extrapolation is shifted per-layer according to an inverted-U normalization
    derived from ``u_norm = 1 - (2*layer_idx/(N-1) - 1)^2``.

    Args:
        dim: Model embedding dimension (must be even; default 128).
        base: Base value for rotary position encoding frequencies (default 10000).
        L: Target maximum sequence length (default 2048).
        alpha: Lower clipping bound for the raw weight ratio (default 1.0).
        beta: Upper clipping bound for the raw weight ratio (default 32.0).
        layer_idx: Index of the current transformer layer (0-based, default 0).
        N: Total number of transformer layers in the model (default 32).

    Returns:
        numpy.ndarray: Array of shape ``(dim // 2,)`` containing extrapolation weights
            for each frequency group. Values near 1.0 indicate extrapolation; values
            near 0.0 indicate interpolation.
    """
    # Compute wavelengths and raw extrapolation ratios for each frequency group
    theta = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    lambda_d = 2 * np.pi / theta
    r_d = L / lambda_d
    w_ext = np.clip((r_d - alpha) / (beta - alpha), 0.0, 1.0)

    # Locate the transition boundaries in the frequency spectrum
    upper_bound = dim // 8 // 2 - 1
    beta_idx = 0
    alpha_idx = 0
    for idx in range(len(w_ext)):
        if w_ext[idx] >= 1.0:
            beta_idx = idx
        if w_ext[idx] > 0.0:
            alpha_idx = idx + 1
    shift = beta_idx - upper_bound

    # Compute inverted-U layer normalization factor
    x = 10 * (layer_idx / (N - 1)) - 5
    l_norm = 1 / (1 + math.exp(x))
    current_shift = int(l_norm * shift)
    layer_norm = 2.0 * layer_idx / (N - 1) - 1.0
    u_norm = 1.0 - layer_norm**2  # Inverted-U shape: peaks at middle layers
    current_shift = int(shift * u_norm)  # More shifting at edge layers

    # Apply layer-dependent shift to the interpolation/extrapolation boundaries
    beta_idx_layer = beta_idx - current_shift
    alpha_idx_layer = alpha_idx - current_shift

    # Build the shifted weight vector
    w_ext_layer = np.empty_like(w_ext)
    w_ext_layer[:beta_idx_layer] = 1.0
    if beta_idx_layer < alpha_idx_layer:
        w_ext_layer[beta_idx_layer:alpha_idx_layer] = w_ext[beta_idx:alpha_idx]
    w_ext_layer[alpha_idx_layer:] = 0.0

    # Apply u_norm-modulated alpha/beta bounds for smoother per-layer variation
    alpha_base, alpha_range = 1.0, 1.0
    beta_base, beta_range = 32.0, 24.0

    alpha = alpha_base + alpha_range * u_norm
    beta = beta_base + beta_range * u_norm
    w_ext_layer = np.clip((r_d - alpha) / (beta - alpha), 0.0, 1.0)

    return w_ext_layer  # Weights for dim//2 frequency groups


if __name__ == "__main__":
    N = 32  # Total number of layers (indices 0~31; 0=shallowest, 31=deepest)
    dim = 128
    freq_group_count = dim // 2  # 64 frequency groups
    # Generate x-axis labels corresponding to original dimension indices (0, 2, 4, ..., 126)
    x_dim_labels = np.arange(0, dim, 2)  # Shape: (64,)
    weights_matrix = np.zeros((N, freq_group_count))  # (32, 64)

    # Calculate per-layer weights
    for layer_idx in range(N):
        w_ext = ntk_by_parts_weights_by_layer(layer_idx=layer_idx, N=N, dim=dim)
        weights_matrix[layer_idx] = w_ext
        print(f"Calculated weights for layer {layer_idx:2d}")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Render heatmap with y=0 at bottom (layer 0) and y=31 at top (layer 31)
    im = ax.imshow(
        weights_matrix,
        cmap="YlOrRd_r",  # Dark red = extrapolation (1), light yellow = interpolation (0)
        aspect="auto",
        extent=[x_dim_labels[0], x_dim_labels[-1], -0.5, N - 0.5],
    )

    # Configure axes labels and title
    ax.set_xlabel("Dimension index", fontsize=14, labelpad=10)
    ax.set_ylabel("Transformer layer index", fontsize=14, labelpad=10)
    ax.set_title("Extrapolation weight heatmap", fontsize=16, pad=20)

    # Set tick positions: x-axis every 20 dimensions, y-axis shows every layer
    ax.set_xticks(np.arange(0, dim, 20))
    ax.set_yticks(np.arange(N))  # Display all layer indices (0~31)
    # Ensure y-axis range places layer 0 at bottom and layer 31 at top
    ax.set_ylim(-0.5, N - 0.5)

    # Add colorbar with descriptive label
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(
        "Extrapolation weight w_ext (1=extrapolation, 0=interpolation)",
        fontsize=12,
        labelpad=10,
    )

    # Save figure to disk
    plt.tight_layout()
    plt.savefig("heatmap_U2_max.png", dpi=300, bbox_inches="tight")
    print("Heatmap saved as: heatmap_U2.png")
