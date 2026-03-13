import math
import numpy as np
import matplotlib.pyplot as plt

# 处理中文和负号显示
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
    theta = 1.0 / (base ** (np.arange(0, dim, 2) / dim))  # 64个频率组（对应维度0,2,...126）
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
    u_norm = 1.0 - layer_norm ** 2  # 倒 U
    current_shift = int(shift * u_norm)  # 两头大，中间小
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

    return w_ext_layer  # 64个频率组的权重


if __name__ == "__main__":
    N = 32  # 总层数（0~31，0=最浅层，31=最深层）
    dim = 128
    freq_group_count = dim // 2  # 64个频率组
    # 生成x轴对应的原始维度（0,2,4,...,126）
    x_dim_labels = np.arange(0, dim, 2)  # 形状：(64,)
    weights_matrix = np.zeros((N, freq_group_count))  # (32, 64)

    # 计算每一层的权重
    for layer_idx in range(N):
        w_ext = ntk_by_parts_weights_by_layer(layer_idx=layer_idx, N=N, dim=dim)
        weights_matrix[layer_idx] = w_ext
        print(f"已计算第 {layer_idx:2d} 层权重")

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制热力图：x轴对应原始维度（0,2,...126），y轴0在下、31在上
    im = ax.imshow(
        weights_matrix,
        cmap="YlOrRd_r",  # 1=深红（外推），0=浅黄（内插）
        aspect="auto",
        # 坐标范围：x从第一个维度到最后一个维度，y从0到31（层索引）
        extent=[x_dim_labels[0], x_dim_labels[-1], -0.5, N - 0.5]
    )

    # 3. 设置坐标轴和标签
    ax.set_xlabel("维度索引", fontsize=14, labelpad=10)
    ax.set_ylabel("Transformer层索引", fontsize=14, labelpad=10)
    ax.set_title("外推权重热力图", fontsize=16, pad=20)

    # 刻度设置：x轴每5个频率组标一次，y轴显示所有层
    ax.set_xticks(np.arange(0, dim, 20))
    ax.set_yticks(np.arange(N))  # 显示所有层索引（0~31）
    ax.set_ylim(-0.5, N-0.5)  # 关键：y轴下限-0.5（层0下方），上限31.5（层31上方），确保0在下、31在上

    # 4. 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("外推权重 w_ext（1=外推，0=内插）", fontsize=12, labelpad=10)

    # 5. 保存图片
    plt.tight_layout()
    plt.savefig("heatmap_U2_max.png", dpi=300, bbox_inches="tight")
    print("热力图已保存为：heatmap_U2.png")