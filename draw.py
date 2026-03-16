import math
import numpy as np
import matplotlib.pyplot as plt

# Parameter settings
N = 32
layer_indices = np.arange(0, 32, 1)  # layer_idx = 0,1,...,31


# Calculate three types of l_norm
def log_func1(layer_idx):
    k = 2.0
    return math.log(1.0 + k * layer_idx) / math.log(1.0 + k * (N - 1))


def log_func2(layer_idx):
    k = 6.0
    return math.log(1.0 + k * layer_idx) / math.log(1.0 + k * (N - 1))


def exp_func1(layer_idx):
    tau = 0.2
    return (math.exp(tau * layer_idx) - 1) / (math.exp(tau * (N - 1)) - 1)


def exp_func2(layer_idx):
    tau = 2.0
    return (math.exp(tau * layer_idx) - 1) / (math.exp(tau * (N - 1)) - 1)


def sigmoid1(layer_idx):
    x = 20 * (layer_idx / 31) - 10
    return 1 / (1 + math.exp(-x))


def sigmoid2(layer_idx):
    x = 20 * (layer_idx / 31) - 10
    return 1 / (1 + math.exp(3 - x))


# Batch calculation
l_norm1 = [log_func1(l) for l in layer_indices]
l_norm2 = [log_func2(l) for l in layer_indices]
l_norm3 = [exp_func1(l) for l in layer_indices]
l_norm4 = [exp_func2(l) for l in layer_indices]
l_norm5 = [sigmoid1(l) for l in layer_indices]
l_norm6 = [sigmoid2(l) for l in layer_indices]

# Plotting
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

# Style settings
plt.xlabel("layer_idx (0~31)", fontsize=12)
plt.ylabel("l_norm", fontsize=12)
plt.title("Comparison of three layer factor functions (N=32)", fontsize=14)
plt.ylim(0, 1.05)
plt.xlim(0, 31)
plt.grid(linestyle="--", alpha=0.7)
plt.legend(fontsize=12)
plt.show()
