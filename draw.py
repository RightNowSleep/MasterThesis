"""Layer normalization function comparison plotter for attention scaling.

Plots six candidate functions (logarithmic, exponential, sigmoid variants)
that map transformer layer indices (0–31) to a normalized range [0, 1].
These functions are used to determine per-layer attention scaling factors
in context extension research.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# Model configuration: 32 transformer layers
N = 32
layer_indices = np.arange(0, 32, 1)  # layer_idx = 0, 1, ..., 31


def log_func1(layer_idx):
    """Calculate normalized layer index using logarithmic function with k=2.0.

    Uses the formula: ``log(1 + k * x) / log(1 + k * (N-1))``, producing a
    concave curve that grows quickly at shallow layers and saturates at deep layers.

    Args:
        layer_idx: The layer index to normalize (0-based).

    Returns:
        float: Normalized value in [0, 1].
    """
    k = 2.0
    return math.log(1.0 + k * layer_idx) / math.log(1.0 + k * (N - 1))


def log_func2(layer_idx):
    """Calculate normalized layer index using logarithmic function with k=6.0.

    Uses the same logarithmic formula as :func:`log_func1` but with a larger
    growth constant, resulting in earlier saturation.

    Args:
        layer_idx: The layer index to normalize (0-based).

    Returns:
        float: Normalized value in [0, 1].
    """
    k = 6.0
    return math.log(1.0 + k * layer_idx) / math.log(1.0 + k * (N - 1))


def exp_func1(layer_idx):
    """Calculate normalized layer index using exponential function with tau=0.2.

    Uses the formula: ``(exp(tau * x) - 1) / (exp(tau * (N-1)) - 1)``, producing
    a gentle convex curve suitable for early-layer emphasis.

    Args:
        layer_idx: The layer index to normalize (0-based).

    Returns:
        float: Normalized value in [0, 1].
    """
    tau = 0.2
    return (math.exp(tau * layer_idx) - 1) / (math.exp(tau * (N - 1)) - 1)


def exp_func2(layer_idx):
    """Calculate normalized layer index using exponential function with tau=2.0.

    Same formula as :func:`exp_func1` but with stronger growth, creating a steep
    curve that emphasizes deeper layers.

    Args:
        layer_idx: The layer index to normalize (0-based).

    Returns:
        float: Normalized value in [0, 1].
    """
    tau = 2.0
    return (math.exp(tau * layer_idx) - 1) / (math.exp(tau * (N - 1)) - 1)


def sigmoid1(layer_idx):
    """Calculate normalized layer index using a standard sigmoid function.

    Maps layer indices through a sigmoid centered at the midpoint, producing an
    S-shaped transition from ~0 to ~1.

    Args:
        layer_idx: The layer index to normalize (0-based).

    Returns:
        float: Normalized value approximately in (0, 1).
    """
    x = 20 * (layer_idx / 31) - 10
    return 1 / (1 + math.exp(-x))


def sigmoid2(layer_idx):
    """Calculate normalized layer index using a right-shifted sigmoid function.

    Applies a horizontal shift to the sigmoid so that the transition occurs
    later (toward deeper layers).

    Args:
        layer_idx: The layer index to normalize (0-based).

    Returns:
        float: Normalized value approximately in (0, 1).
    """
    x = 20 * (layer_idx / 31) - 10
    return 1 / (1 + math.exp(3 - x))


# Compute normalization values for all layers under each function
l_norm1 = [log_func1(l) for l in layer_indices]
l_norm2 = [log_func2(l) for l in layer_indices]
l_norm3 = [exp_func1(l) for l in layer_indices]
l_norm4 = [exp_func2(l) for l in layer_indices]
l_norm5 = [sigmoid1(l) for l in layer_indices]
l_norm6 = [sigmoid2(l) for l in layer_indices]

# Plot all six curves on a single figure
plt.figure(figsize=(10, 6))
plt.plot(layer_indices, l_norm1, "o-", label="log (k=2.0)", color="blue", markersize=4)
plt.plot(layer_indices, l_norm2, "o-", label="log (k=6.0)", color="green", markersize=4)
plt.plot(layer_indices, l_norm3, "s-", label="exp (t=0.2)", color="red", markersize=4)
plt.plot(
    layer_indices,
    l_norm4,
    "s-",
    label="exp (t=0.6)",
    color="purple",
    markersize=4,
)
plt.plot(layer_indices, l_norm5, "*-", label="sigmoid", color="orange", markersize=4)
plt.plot(layer_indices, l_norm6, "*-", label="sigmoid", color="yellow", markersize=4)

# Apply style settings
plt.xlabel("layer_idx (0~31)", fontsize=12)
plt.ylabel("l_norm", fontsize=12)
plt.title("Comparison of three layer factor functions (N=32)", fontsize=14)
plt.ylim(0, 1.05)
plt.xlim(0, 31)
plt.grid(linestyle="--", alpha=0.7)
plt.legend(fontsize=12)
plt.show()
