import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def calculate_u_norm(layer, N):
    """
    根据公式计算u_norm
    公式5: x = (2*layer)/(N-1) - 1
    公式6: u_norm = 1 - x^2
    """
    x = (2 * layer) / (N - 1) - 1
    u_norm = 1 - x**2
    return u_norm

def attention_scale_factor(u_norm, S):
    """
    计算注意力缩放系数
    u_norm: 根据公式5-6计算的归一化位置
    S: 上下文长度扩展比例 L_ext/L
    """
    return 1 + 0.1 * (1 - u_norm) * np.log(S)

# 模拟一个有32层的模型
N = 32
layers = np.arange(N)
u_norm_values = calculate_u_norm(layers, N)
S_range = np.linspace(1.1, 32, 100)

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 子图1: u_norm随网络层的变化
axes[0,0].plot(layers, u_norm_values, 'b-', linewidth=2, label='u_norm')
axes[0,0].set_xlabel('网络层索引')
axes[0,0].set_ylabel('u_norm')
axes[0,0].set_title('u_norm 随网络层的变化')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].axvline(x=0, color='red', linestyle=':', alpha=0.7, label='浅层')
axes[0,0].axvline(x=N//2, color='orange', linestyle='--', alpha=0.7, label='中间层')
axes[0,0].axvline(x=N-1, color='green', linestyle='-.', alpha=0.7, label='深层')
axes[0,0].legend()

# 子图2: 不同S值下缩放系数随网络层的变化
S_values = [2, 4, 8, 16, 32]
colors = plt.cm.viridis(np.linspace(0, 1, len(S_values)))

for i, s in enumerate(S_values):
    scale_factors = attention_scale_factor(u_norm_values, s)
    axes[0,1].plot(layers, scale_factors, label=f'S={s}', color=colors[i], linewidth=2)

axes[0,1].set_xlabel('网络层索引')
axes[0,1].set_ylabel('注意力缩放系数 √t')
axes[0,1].set_title('注意力缩放系数随网络层的变化')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)
axes[0,1].axvline(x=0, color='red', linestyle=':', alpha=0.7, label='浅层')
axes[0,1].axvline(x=N//2, color='orange', linestyle='--', alpha=0.7, label='中间层')
axes[0,1].axvline(x=N-1, color='green', linestyle='-.', alpha=0.7, label='深层')

# 子图3: u_norm和S对缩放系数的影响（区分浅层和深层）
middle_layer_idx = N // 2
shallow_layer_idx = 0
deep_layer_idx = N - 1

u_norm_middle = u_norm_values[middle_layer_idx]
u_norm_shallow = u_norm_values[shallow_layer_idx]
u_norm_deep = u_norm_values[deep_layer_idx]

scale_middle = attention_scale_factor(u_norm_middle, S_range)
scale_shallow = attention_scale_factor(u_norm_shallow, S_range)
scale_deep = attention_scale_factor(u_norm_deep, S_range)

# 由于浅层和深层的u_norm相同，缩放系数曲线完全重叠，使用相同颜色但不同线型来区分概念
axes[1,0].plot(S_range, scale_middle, label=f'中间层 (u_norm={u_norm_middle:.3f})',
               linewidth=2, color='orange', linestyle='-')
# 绘制浅层和深层曲线（它们完全重叠）
axes[1,0].plot(S_range, scale_shallow, label=f'浅层/深层 (u_norm={u_norm_shallow:.3f})',
               linewidth=2, color='red', linestyle='--')

# 添加点来标记特定S值下的缩放系数
S_specific = [4, 8, 16]
scale_shallow_specific = attention_scale_factor(u_norm_shallow, np.array(S_specific))
axes[1,0].scatter(S_specific, scale_shallow_specific,
                 c=['red'], s=80, zorder=5,
                 label='浅层/深层示例点',
                 edgecolors='white', linewidth=1)
axes[1,0].set_xlabel('上下文扩展比例 S')
axes[1,0].set_ylabel('注意力缩放系数 √t')
axes[1,0].set_title('不同网络层的缩放系数随S的变化')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 添加文本注释说明浅层和深层的对称性
axes[1,0].text(0.02, 0.98, f'注意：浅层(layer={shallow_layer_idx})和深层(layer={deep_layer_idx})\n具有相同的u_norm值({u_norm_shallow:.3f})，因此缩放系数曲线完全重叠',
               transform=axes[1,0].transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


# 子图4: 3D可视化 - u_norm vs S vs 缩放系数
u_norm_range = np.linspace(0, 1, 100)
U_norm_mesh, S_mesh = np.meshgrid(u_norm_range, S_range)
T_mesh = attention_scale_factor(U_norm_mesh, S_mesh)

im = axes[1,1].contourf(U_norm_mesh, S_mesh, T_mesh, levels=20, cmap='viridis')
axes[1,1].set_xlabel('u_norm')
axes[1,1].set_ylabel('上下文扩展比例 S')
axes[1,1].set_title('注意力缩放系数等高线图')
plt.colorbar(im, ax=axes[1,1])

# 在等高线图上标注实际的u_norm值
axes[1,1].scatter([u_norm_shallow, u_norm_middle, u_norm_deep], [2, 2, 2],
                 c=['red', 'orange', 'green'], s=100, zorder=5,
                 label='实际层位置', edgecolors='white', linewidth=1)
axes[1,1].legend()

plt.tight_layout()
plt.savefig("attention_scaling.png", dpi=300, bbox_inches='tight')

# 打印公式和分析
print("公式5: x = (2*layer)/(N-1) - 1")
print("公式6: u_norm = 1 - x²")
print("缩放系数公式: √t = 1 + 0.1 * (1 - u_norm) * log(S)")
print()
print(f"对于{N}层模型:")
print(f"- 浅层(layer=0): x={(2*0)/(N-1)-1:.3f}, u_norm={calculate_u_norm(0, N):.3f}")
print(f"- 中间层(layer={N//2}): x={(2*(N//2))/(N-1)-1:.3f}, u_norm={calculate_u_norm(N//2, N):.3f}")
print(f"- 深层(layer={N-1}): x={(2*(N-1))/(N-1)-1:.3f}, u_norm={calculate_u_norm(N-1, N):.3f}")
print()
print("注意: 根据公式，中间层u_norm最大，浅层和深层u_norm最小")
print("这导致(1-u_norm)在中间层最小，在浅层和深层最大")
print("因此缩放系数在中间层最小，在浅层和深层最大")
print("这与论文中'中间层需要较多注意力缩放'的描述似乎矛盾")
print("可能需要重新审视公式与实际需求的对应关系")



