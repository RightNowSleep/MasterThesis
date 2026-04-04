"""Block size sum visualization for Freq-Reciprocal position encoding analysis.

This module computes and visualizes the total sum of dimension-wise block sizes
(sum of b_i) used in the Freq-Reciprocal (FreqReciprocal) Rotary Position
Embedding method as a function of the context extension ratio S.

The Freq-Reciprocal method partitions RoPE dimensions into two regions based on
a threshold index i_star:

    - Dimensions i < i_star: Receive linearly interpolated scaling factors
      that transition smoothly from 1.0 to S.
    - Dimensions i >= i_star: Receive the full scaling factor S.

This script calculates the mean block size across all dimensions for a range of
S values and plots the result to illustrate how the total scaling overhead grows
with the extension ratio. The output is useful for understanding the trade-off
between context length extension and positional resolution preservation.

Output:
    drawer/block_sum.png: Line plot showing log-sum of block sizes vs S at 600 DPI.
    Console table: Key data points at S = 2, 4, 8, 16, 32.
"""

import math
import matplotlib.pyplot as plt
import numpy as np


def calculate_block_sum(S, d=128, L=2048, base=10000):
    """Calculate the logarithm of the mean block size sum across all dimensions.

    Follows the exact logic of LlamaFreqReciprocalRotaryEmbedding._compute_block_sizes,
    computing each dimension's b_i step by step without using the optimized closed-form
    summation formula. The result is the log of the arithmetic mean of all b_i values.

    Algorithm:
        1. Determine the threshold dimension index i_star where the rotation count
           r_i drops below 1.0 within the original context length L.
        2. Compute the slope K from the boundary conditions: b_0 = 1.0, b_{i_star} = S.
        3. For each dimension i < i_star: b_i = 1 + K * (inv_theta_i - 1), clamped to [1, S].
        4. For each dimension i >= i_star: b_i = S.
        5. Return the natural logarithm of the mean of all b_i values.

    Args:
        S (float): Context extension ratio. Must be greater than 1.0 to trigger scaling.
            When S <= 1.0, all blocks default to 1.0 (no extension).
        d (int, optional): Full RoPE dimension size (must be even). Defaults to 128.
        L (int, optional): Original maximum position embedding length in tokens.
            Defaults to 2048.
        base (float, optional): Base frequency for computing inverse frequencies.
            Defaults to 10000.

    Returns:
        float: Natural logarithm of the mean block size sum (log(mean(sum(b_i)))).
            This log transform compresses the dynamic range for clearer visualization.
    """
    if S <= 1.0:
        return d // 2 * 1.0

    N = d // 2

    # Find the threshold dimension i_star where rotation count r_i first drops below 1.0
    i_star = N

    for i in range(N):
        theta_i = base ** (-2.0 * i / d)
        r_i = L * theta_i / (2.0 * math.pi)
        if r_i < 1.0:
            i_star = i
            break

    # Compute the slope K from boundary conditions at i_star
    inv_theta_istar = base ** (2.0 * i_star / d)

    denom = inv_theta_istar - 1.0
    if abs(denom) < 1e-8:
        K = 0.0
    else:
        K = (S - 1.0) / denom

    # Accumulate the mean block size across all dimensions
    total_sum = 0.0

    for i in range(N):
        if i < i_star:
            inv_theta_i = base ** (2.0 * i / d)
            b_i = 1.0 + K * (inv_theta_i - 1.0)
        else:
            b_i = S

        # Clamp each block size to the valid range [1.0, S]
        b_i = max(1.0, min(b_i, S))

        total_sum += b_i / N

    return math.log(total_sum)


# Generate data points across the range of extension ratios
S_values = np.linspace(1.1, 32, 100)
sums = []

for S in S_values:
    val = calculate_block_sum(S)
    sums.append(val)

# Print summary table for key extension ratios
print(f"{'Extension Ratio (S)':<20} | {'Total Block Sum':<20}")
print("-" * 45)
for S in [2, 4, 8, 16, 32]:
    val = calculate_block_sum(S)
    print(f"{S:<20} | {val:.4f}")

# Generate visualization plot
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

# Annotate key data points on the plot
for S in [2, 4, 8, 16, 32]:
    val = calculate_block_sum(S)
    plt.scatter(S, val, color="red", zorder=5)
    plt.text(S, val + 1.0, f"S={S}\n{val:.1f}", ha="center", fontsize=9)

plt.title(
    "Total Sum of Block Sizes with Context Extension Ratio S (L=2k, d=128)", fontsize=14
)
plt.xlabel("Extension Ratio (S)", fontsize=12)
plt.ylabel("Sum of Block Sizes ($\\sum b_i$)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.xticks(np.arange(0, 34, 2))

plt.savefig("drawer/block_sum.png", dpi=600)
