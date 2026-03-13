import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 准备数据
methods = ['YaRN', '本文方法']
datasets = ['ARC_c', 'Hellaswag', 'MMLU']
yarn_values = [39.85, 54.52, 24.18]
my_method_values = [41.04, 54.94, 25.17]

# 设置柱状图参数
x = np.arange(len(datasets))  # 数据集位置
width = 0.35  # 柱子宽度

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))

# 创建柱状图
bars1 = ax.bar(x - width/2, yarn_values, width, label='YaRN', color='#1f77b4', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, my_method_values, width, label='本文方法', color='#ff7f0e', alpha=0.8, edgecolor='black')

# 添加数据标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold')

add_labels(bars1)
add_labels(bars2)

# 设置图表标题和标签
ax.set_title('不同上下文扩展方法在64k上下文窗口下的性能对比', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('评估数据集', fontsize=14, labelpad=10)
ax.set_ylabel('准确率 (%)', fontsize=14, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12)
ax.legend(fontsize=12, loc='upper right')

# 设置y轴范围，以便更清晰地显示差异
min_val = min(min(yarn_values), min(my_method_values)) - 2
max_val = max(max(yarn_values), max(my_method_values)) + 2
ax.set_ylim(min_val, max_val)

# 添加网格线
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 添加性能差异标注
for i in range(len(datasets)):
    diff = my_method_values[i] - yarn_values[i]
    if diff > 0:
        ax.annotate(f'+{diff:.2f}%',
                   xy=(x[i], max(yarn_values[i], my_method_values[i]) + 0.5),
                   ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8),
                   fontweight='bold', color='red')

# 优化布局
plt.tight_layout()

plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')

print("图表已生成，展示了两种方法在三个数据集上的性能对比。")
print("本文方法在所有数据集上均优于YaRN方法：")
for i, dataset in enumerate(datasets):
    diff = my_method_values[i] - yarn_values[i]
    print(f"- {dataset}: 提升 {diff:.2f} 个百分点")