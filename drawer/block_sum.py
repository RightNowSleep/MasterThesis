import math
import matplotlib.pyplot as plt
import numpy as np


def calculate_block_sum(S, d=128, L=2048, base=10000):
    """
    计算所有维度上的块大小总和 sum(b_i)。

    严格按照 LlamaFreqReciprocalRotaryEmbedding._compute_block_sizes 的逻辑，
    逐步计算每个维度的 b_i，不使用求和公式的优化形式。
    """
    if S <= 1.0:
        return d // 2 * 1.0

    N = d // 2  # 维度数量

    # 1. 计算临界索引 i*
    # r_i = L * theta_i / (2pi)
    # theta_i = base^(-2i/d)
    # 找到第一个 r_i < 1 的索引
    i_star = N  # 默认值，如果所有维度都需要插值

    for i in range(N):
        theta_i = base ** (-2.0 * i / d)
        r_i = L * theta_i / (2.0 * math.pi)
        if r_i < 1.0:
            i_star = i
            break

    # 2. 计算 1/θ_{i*}
    inv_theta_istar = base ** (2.0 * i_star / d)

    # 3. 计算归一化常数 K
    denom = inv_theta_istar - 1.0
    if abs(denom) < 1e-8:
        K = 0.0
    else:
        K = (S - 1.0) / denom

    # 4. 逐步求和计算
    total_sum = 0.0

    for i in range(N):
        if i < i_star:
            # b_i = 1 + K * (1/θ_i - 1)
            inv_theta_i = base ** (2.0 * i / d)
            b_i = 1.0 + K * (inv_theta_i - 1.0)
        else:
            # b_i = S
            b_i = S

        # 进行 clamp 处理，确保在 [1, S] 范围内
        b_i = max(1.0, min(b_i, S))

        total_sum += b_i / N

    return math.log(total_sum)


# ==========================================
# 绘图部分
# ==========================================

# 设置 S 的取值范围
S_values = np.linspace(1.1, 32, 100)
sums = []

for S in S_values:
    val = calculate_block_sum(S)
    sums.append(val)

# 打印关键点的数值
print(f"{'Extension Ratio (S)':<20} | {'Total Block Sum':<20}")
print("-" * 45)
for S in [2, 4, 8, 16, 32]:
    val = calculate_block_sum(S)
    print(f"{S:<20} | {val:.4f}")

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(
    S_values,
    sums,
    color="royalblue",
    linewidth=2,
    marker="o",
    markersize=4,
    label="Sum of Block Sizes",
)

# 标注关键点
for S in [2, 4, 8, 16, 32]:
    val = calculate_block_sum(S)
    plt.scatter(S, val, color="red", zorder=5)
    plt.text(S, val + 1.0, f"S={S}\n{val:.1f}", ha="center", fontsize=9)

plt.title(
    "Total Sum of Block Sizes with Context Extension Ratio S (L=2k, d=128)", fontsize=14
)
plt.xlabel("Extension Ratio (S)", fontsize=12)
plt.ylabel("Sum of Block Sizes ($\sum b_i$)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.xticks(np.arange(0, 34, 2))

plt.savefig("drawer/block_sum.png", dpi=600)
