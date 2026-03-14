# sinusoidal_pe_heatmap_with_box.py
import numpy as np
import matplotlib.pyplot as plt

# ========== 参数 ==========
d_model = 128
max_len = 4096
base = 10000.0

# ========== 构造 PE ==========
pe = np.zeros((max_len, d_model))
dim_vec = np.arange(0, d_model, 2).reshape(1, -1)
angle = np.arange(max_len).reshape(-1, 1) / (base ** (dim_vec / d_model))

pe[:, 0::2] = np.sin(angle)
pe[:, 1::2] = np.cos(angle)

# ========== 绘图 ==========
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(pe, cmap='viridis', aspect='auto')

# 红色框：≥2048 的区域
start_row = 2048
box_height = max_len - start_row
ax.add_patch(plt.Rectangle((-0.5, start_row - 0.5),   # 左上角 (x, y)
                           d_model, box_height,       # 宽、高
                           fill=False, edgecolor='red', lw=5))

plt.colorbar(im, label='PE value')
plt.title('Position Encoding (red box: position ≥ 2048)')
plt.xlabel('dimension')
plt.ylabel('position')
plt.tight_layout()
plt.savefig("heatmap_position_encoding.png", dpi=300)
