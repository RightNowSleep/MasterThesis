import matplotlib.pyplot as plt
import numpy as np

# Set Chinese font for matplotlib
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Prepare data
methods = ["YaRN", "Our Method"]
datasets = ["ARC_c", "Hellaswag", "MMLU"]
yarn_values = [39.85, 54.52, 24.18]
my_method_values = [41.04, 54.94, 25.17]

# Set bar chart parameters
x = np.arange(len(datasets))  # Dataset positions
width = 0.35  # Bar width

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create bar chart
bars1 = ax.bar(
    x - width / 2,
    yarn_values,
    width,
    label="YaRN",
    color="#1f77b4",
    alpha=0.8,
    edgecolor="black",
)
bars2 = ax.bar(
    x + width / 2,
    my_method_values,
    width,
    label="Our Method",
    color="#ff7f0e",
    alpha=0.8,
    edgecolor="black",
)


# Add data labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )


add_labels(bars1)
add_labels(bars2)

# Set chart title and labels
ax.set_title(
    "Performance Comparison of Different Context Extension Methods at 64k Context Window",
    fontsize=16,
    fontweight="bold",
    pad=20,
)
ax.set_xlabel("Evaluation Dataset", fontsize=14, labelpad=10)
ax.set_ylabel("Accuracy (%)", fontsize=14, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12)
ax.legend(fontsize=12, loc="upper right")

# Set y-axis range for clearer difference display
min_val = min(min(yarn_values), min(my_method_values)) - 2
max_val = max(max(yarn_values), max(my_method_values)) + 2
ax.set_ylim(min_val, max_val)

# Add grid lines
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Add performance difference annotations
for i in range(len(datasets)):
    diff = my_method_values[i] - yarn_values[i]
    if diff > 0:
        ax.annotate(
            f"+{diff:.2f}%",
            xy=(x[i], max(yarn_values[i], my_method_values[i]) + 0.5),
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8),
            fontweight="bold",
            color="red",
        )

# Optimize layout
plt.tight_layout()

plt.savefig("performance_comparison.png", dpi=300, bbox_inches="tight")

print(
    "Chart generated, showing performance comparison of two methods across three datasets."
)
print("Our method outperforms YaRN on all datasets:")
for i, dataset in enumerate(datasets):
    diff = my_method_values[i] - yarn_values[i]
    print(f"- {dataset}: improvement of {diff:.2f} percentage points")
