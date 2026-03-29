import math
import matplotlib.pyplot as plt
import numpy as np


def calculate_block_sum(S, d=128, L=2048, base=10000):
    """Calculate the sum of block sizes across all dimensions sum(b_i).

    Follows the exact logic of LlamaFreqReciprocalRotaryEmbedding._compute_block_sizes,
    computing each dimension's b_i step by step without using the optimized summation formula.

    Args:
        S: Context extension ratio, should be greater than 1.0.
        d: RoPE dimension size. Defaults to 128.
        L: Original maximum position embedding length. Defaults to 2048.
        base: RoPE base frequency. Defaults to 10000.

    Returns:
        Logarithm of the total sum of block sizes across all dimensions.
    """
    if S <= 1.0:
        return d // 2 * 1.0

    N = d // 2

    i_star = N

    for i in range(N):
        theta_i = base ** (-2.0 * i / d)
        r_i = L * theta_i / (2.0 * math.pi)
        if r_i < 1.0:
            i_star = i
            break

    inv_theta_istar = base ** (2.0 * i_star / d)

    denom = inv_theta_istar - 1.0
    if abs(denom) < 1e-8:
        K = 0.0
    else:
        K = (S - 1.0) / denom

    total_sum = 0.0

    for i in range(N):
        if i < i_star:
            inv_theta_i = base ** (2.0 * i / d)
            b_i = 1.0 + K * (inv_theta_i - 1.0)
        else:
            b_i = S

        b_i = max(1.0, min(b_i, S))

        total_sum += b_i / N

    return math.log(total_sum)


S_values = np.linspace(1.1, 32, 100)
sums = []

for S in S_values:
    val = calculate_block_sum(S)
    sums.append(val)

print(f"{'Extension Ratio (S)':<20} | {'Total Block Sum':<20}")
print("-" * 45)
for S in [2, 4, 8, 16, 32]:
    val = calculate_block_sum(S)
    print(f"{S:<20} | {val:.4f}")

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
