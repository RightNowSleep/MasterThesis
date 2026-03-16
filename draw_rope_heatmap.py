import math
import numpy as np
import matplotlib.pyplot as plt

# Handle Chinese and minus sign display
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
    # 64 frequency groups (corresponding to dimensions 0,2,...126)
    theta = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    lambda_d = 2 * np.pi / theta
    r_d = L / lambda_d
    w_ext = np.clip((r_d - alpha) / (beta - alpha), 0.0, 1.0)

    upper_bound = dim // 8 // 2 - 1
    beta_idx = 0
    alpha_idx = 0
    for idx in range(len(w_ext)):
        if w_ext[idx] >= 1.0:
            beta_idx = idx
        if w_ext[idx] > 0.0:
            alpha_idx = idx + 1
    shift = beta_idx - upper_bound
    x = 10 * (layer_idx / (N - 1)) - 5
    l_norm = 1 / (1 + math.exp(x))
    current_shift = int(l_norm * shift)
    layer_norm = 2.0 * layer_idx / (N - 1) - 1.0
    u_norm = 1.0 - layer_norm**2  # Inverted U shape
    current_shift = int(shift * u_norm)  # Large at both ends, small in the middle
    beta_idx_layer = beta_idx - current_shift
    alpha_idx_layer = alpha_idx - current_shift

    w_ext_layer = np.empty_like(w_ext)
    w_ext_layer[:beta_idx_layer] = 1.0
    if beta_idx_layer < alpha_idx_layer:
        w_ext_layer[beta_idx_layer:alpha_idx_layer] = w_ext[beta_idx:alpha_idx]
    w_ext_layer[alpha_idx_layer:] = 0.0

    alpha_base, alpha_range = 1.0, 1.0
    beta_base, beta_range = 32.0, 24.0

    alpha = alpha_base + alpha_range * u_norm
    beta = beta_base + beta_range * u_norm
    w_ext_layer = np.clip((r_d - alpha) / (beta - alpha), 0.0, 1.0)

    return w_ext_layer  # Weights for 64 frequency groups


if __name__ == "__main__":
    N = 32  # Total number of layers (0~31, 0=shallowest, 31=deepest)
    dim = 128
    freq_group_count = dim // 2  # 64 frequency groups
    # Generate x-axis corresponding to original dimensions (0,2,4,...,126)
    x_dim_labels = np.arange(0, dim, 2)  # Shape: (64,)
    weights_matrix = np.zeros((N, freq_group_count))  # (32, 64)

    # Calculate weights for each layer
    for layer_idx in range(N):
        w_ext = ntk_by_parts_weights_by_layer(layer_idx=layer_idx, N=N, dim=dim)
        weights_matrix[layer_idx] = w_ext
        print(f"Calculated weights for layer {layer_idx:2d}")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot heatmap: x-axis corresponds to original dimensions (0,2,...126), y-axis 0 at bottom, 31 at top
    im = ax.imshow(
        weights_matrix,
        cmap="YlOrRd_r",  # 1=dark red (extrapolation), 0=light yellow (interpolation)
        aspect="auto",
        # Coordinate range: x from first dimension to last dimension, y from 0 to 31 (layer index)
        extent=[x_dim_labels[0], x_dim_labels[-1], -0.5, N - 0.5],
    )

    # 3. Set axes and labels
    ax.set_xlabel("Dimension index", fontsize=14, labelpad=10)
    ax.set_ylabel("Transformer layer index", fontsize=14, labelpad=10)
    ax.set_title("Extrapolation weight heatmap", fontsize=16, pad=20)

    # Tick settings: x-axis labeled every 5 frequency groups, y-axis shows all layers
    ax.set_xticks(np.arange(0, dim, 20))
    ax.set_yticks(np.arange(N))  # Show all layer indices (0~31)
    # Key: y-axis lower limit -0.5 (below layer 0), upper limit 31.5 (above layer 31), ensuring 0 at bottom, 31 at top
    ax.set_ylim(-0.5, N - 0.5)

    # 4. Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(
        "Extrapolation weight w_ext (1=extrapolation, 0=interpolation)",
        fontsize=12,
        labelpad=10,
    )

    # 5. Save figure
    plt.tight_layout()
    plt.savefig("heatmap_U2_max.png", dpi=300, bbox_inches="tight")
    print("Heatmap saved as: heatmap_U2.png")
