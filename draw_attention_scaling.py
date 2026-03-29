import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set Chinese font
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def calculate_u_norm(layer, N):
    """Calculate the normalized layer position factor.

    Computes u_norm using a quadratic formula that creates an inverted U-shape
    pattern across network layers, where middle layers have the highest values.

    Args:
        layer: The layer index (0 to N-1).
        N: Total number of layers in the network.

    Returns:
        float: The normalized layer position factor u_norm, ranging from 0.0
            (at shallow and deep layers) to 1.0 (at middle layers).
    """
    x = (2 * layer) / (N - 1) - 1
    u_norm = 1 - x**2
    return u_norm


def attention_scale_factor(u_norm, S):
    """Calculate the attention scaling factor based on layer position and context extension.

    Computes a scaling factor for attention scores that adjusts based on the layer's
    position in the network and the context length extension ratio.

    Args:
        u_norm: Normalized layer position factor calculated from calculate_u_norm.
            Higher values indicate middle layers, lower values indicate shallow/deep layers.
        S: Context length extension ratio (L_ext / L), where L is the original
            context length and L_ext is the extended context length.

    Returns:
        float: The attention scaling factor √t, where values greater than 1.0
            indicate increased attention scaling for longer contexts.
    """
    return 1 + 0.1 * (1 - u_norm) * np.log(S)


# Simulate a model with 32 layers
N = 32
layers = np.arange(N)
u_norm_values = calculate_u_norm(layers, N)
S_range = np.linspace(1.1, 32, 100)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: u_norm variation across network layers
axes[0, 0].plot(layers, u_norm_values, "b-", linewidth=2, label="u_norm")
axes[0, 0].set_xlabel("Network layer index")
axes[0, 0].set_ylabel("u_norm")
axes[0, 0].set_title("u_norm variation across network layers")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=0, color="red", linestyle=":", alpha=0.7, label="Shallow layer")
axes[0, 0].axvline(
    x=N // 2,
    color="orange",
    linestyle="--",
    alpha=0.7,
    label="Middle layer",
)
axes[0, 0].axvline(
    x=N - 1,
    color="green",
    linestyle="-.",
    alpha=0.7,
    label="Deep layer",
)
axes[0, 0].legend()

# Subplot 2: Scaling factor variation across network layers for different S values
S_values = [2, 4, 8, 16, 32]
colors = plt.cm.viridis(np.linspace(0, 1, len(S_values)))

for i, s in enumerate(S_values):
    scale_factors = attention_scale_factor(u_norm_values, s)
    axes[0, 1].plot(layers, scale_factors, label=f"S={s}", color=colors[i], linewidth=2)

axes[0, 1].set_xlabel("Network layer index")
axes[0, 1].set_ylabel("Attention scaling factor √t")
axes[0, 1].set_title("Attention scaling factor variation across network layers")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=0, color="red", linestyle=":", alpha=0.7, label="Shallow layer")
axes[0, 1].axvline(
    x=N // 2,
    color="orange",
    linestyle="--",
    alpha=0.7,
    label="Middle layer",
)
axes[0, 1].axvline(
    x=N - 1,
    color="green",
    linestyle="-.",
    alpha=0.7,
    label="Deep layer",
)

# Subplot 3: Effect of u_norm and S on scaling factor (distinguishing shallow and deep layers)
middle_layer_idx = N // 2
shallow_layer_idx = 0
deep_layer_idx = N - 1

u_norm_middle = u_norm_values[middle_layer_idx]
u_norm_shallow = u_norm_values[shallow_layer_idx]
u_norm_deep = u_norm_values[deep_layer_idx]

scale_middle = attention_scale_factor(u_norm_middle, S_range)
scale_shallow = attention_scale_factor(u_norm_shallow, S_range)
scale_deep = attention_scale_factor(u_norm_deep, S_range)

# Since shallow and deep layers have the same u_norm, scaling factor curves completely overlap, use same color but different line styles to distinguish concepts
axes[1, 0].plot(
    S_range,
    scale_middle,
    label=f"Middle layer (u_norm={u_norm_middle:.3f})",
    linewidth=2,
    color="orange",
    linestyle="-",
)
# Plot shallow and deep layer curves (they completely overlap)
axes[1, 0].plot(
    S_range,
    scale_shallow,
    label=f"Shallow/Deep layer (u_norm={u_norm_shallow:.3f})",
    linewidth=2,
    color="red",
    linestyle="--",
)

# Add points to mark scaling factor at specific S values
S_specific = [4, 8, 16]
scale_shallow_specific = attention_scale_factor(u_norm_shallow, np.array(S_specific))
axes[1, 0].scatter(
    S_specific,
    scale_shallow_specific,
    c=["red"],
    s=80,
    zorder=5,
    label="Shallow/Deep layer example points",
    edgecolors="white",
    linewidth=1,
)
axes[1, 0].set_xlabel("Context extension ratio S")
axes[1, 0].set_ylabel("Attention scaling factor √t")
axes[1, 0].set_title("Scaling factor variation with S for different network layers")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Add text annotation explaining symmetry between shallow and deep layers
axes[1, 0].text(
    0.02,
    0.98,
    f"Note: Shallow layer (layer={shallow_layer_idx}) and deep layer (layer={deep_layer_idx})\nhave the same u_norm value ({u_norm_shallow:.3f}), so scaling factor curves completely overlap",
    transform=axes[1, 0].transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)


# Subplot 4: 3D visualization - u_norm vs S vs scaling factor
u_norm_range = np.linspace(0, 1, 100)
U_norm_mesh, S_mesh = np.meshgrid(u_norm_range, S_range)
T_mesh = attention_scale_factor(U_norm_mesh, S_mesh)

im = axes[1, 1].contourf(U_norm_mesh, S_mesh, T_mesh, levels=20, cmap="viridis")
axes[1, 1].set_xlabel("u_norm")
axes[1, 1].set_ylabel("Context extension ratio S")
axes[1, 1].set_title("Attention scaling factor contour plot")
plt.colorbar(im, ax=axes[1, 1])

# Mark actual u_norm values on contour plot
axes[1, 1].scatter(
    [u_norm_shallow, u_norm_middle, u_norm_deep],
    [2, 2, 2],
    c=["red", "orange", "green"],
    s=100,
    zorder=5,
    label="Actual layer positions",
    edgecolors="white",
    linewidth=1,
)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("attention_scaling.png", dpi=300, bbox_inches="tight")

# Print formulas and analysis
print("Formula 5: x = (2*layer)/(N-1) - 1")
print("Formula 6: u_norm = 1 - x²")
print("Scaling factor formula: √t = 1 + 0.1 * (1 - u_norm) * log(S)")
print()
print(f"For a {N}-layer model:")
print(
    f"- Shallow layer (layer=0): x={(2*0)/(N-1)-1:.3f}, u_norm={calculate_u_norm(0, N):.3f}"
)
print(
    f"- Middle layer (layer={N//2}): x={(2*(N//2))/(N-1)-1:.3f}, u_norm={calculate_u_norm(N//2, N):.3f}"
)
print(
    f"- Deep layer (layer={N-1}): x={(2*(N-1))/(N-1)-1:.3f}, u_norm={calculate_u_norm(N-1, N):.3f}"
)
print()
print(
    "Note: According to the formula, middle layer u_norm is maximum, shallow and deep layer u_norm is minimum"
)
print(
    "This leads to (1-u_norm) being minimum at middle layer, maximum at shallow and deep layers"
)
print(
    "Therefore scaling factor is minimum at middle layer, maximum at shallow and deep layers"
)
print(
    "This seems to contradict the paper's description that 'middle layers need more attention scaling'"
)
print(
    "May need to re-examine the correspondence between formula and actual requirements"
)
