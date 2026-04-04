# sinusoidal_pe_heatmap_with_box.py
"""Sinusoidal positional encoding heatmap visualizer.

Constructs and plots the standard sinusoidal positional encoding (PE) matrix
used in the original Transformer architecture (Vaswani et al., 2017). Each row
corresponds to a position and each column to a dimension, with alternating
sin/cos patterns across dimension pairs.

A red rectangle highlights positions beyond the original training context length
(default 2048), indicating the extrapolation region where encodings may degrade.
"""

import numpy as np
import matplotlib.pyplot as plt

# ========== Parameters ==========
d_model = 128  # Embedding dimension
max_len = 4096  # Maximum sequence length to visualize
base = 10000.0  # PE base frequency

# ========== Construct positional encoding matrix ==========
pe = np.zeros((max_len, d_model))
dim_vec = np.arange(0, d_model, 2).reshape(1, -1)
angle = np.arange(max_len).reshape(-1, 1) / (base ** (dim_vec / d_model))

pe[:, 0::2] = np.sin(angle)  # Even dimensions: sine
pe[:, 1::2] = np.cos(angle)  # Odd dimensions: cosine

# ========== Plotting ==========
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(pe, cmap="viridis", aspect="auto")

# Red box highlighting the extrapolation region (positions >= 2048)
start_row = 2048
box_height = max_len - start_row
ax.add_patch(
    plt.Rectangle(
        (-0.5, start_row - 0.5),  # Top-left corner (x, y)
        d_model,
        box_height,  # Width, height
        fill=False,
        edgecolor="red",
        lw=5,
    )
)

plt.colorbar(im, label="PE value")
plt.title("Position Encoding (red box: position >= 2048)")
plt.xlabel("dimension")
plt.ylabel("position")
plt.tight_layout()
plt.savefig("heatmap_position_encoding.png", dpi=300)
