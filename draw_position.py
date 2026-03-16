# sinusoidal_pe_heatmap_with_box.py
import numpy as np
import matplotlib.pyplot as plt

# ========== Parameters ==========
d_model = 128
max_len = 4096
base = 10000.0

# ========== Construct PE ==========
pe = np.zeros((max_len, d_model))
dim_vec = np.arange(0, d_model, 2).reshape(1, -1)
angle = np.arange(max_len).reshape(-1, 1) / (base ** (dim_vec / d_model))

pe[:, 0::2] = np.sin(angle)
pe[:, 1::2] = np.cos(angle)

# ========== Plotting ==========
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(pe, cmap="viridis", aspect="auto")

# Red box: region >= 2048
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
