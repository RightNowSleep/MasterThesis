# 单参数效果优先搜索规范

## Why

用户澄清了约束条件：无论上下文窗口多大，只要是在原始上下文窗口内的token，注意力缩放必须严格等于1.0。这意味着 `attn_scale_base` 应该固定为 1.0，不需要搜索。同时，搜索模式应该以效果优先，在保证找到最优参数的前提下尽可能高效。

## What Changes

* **固定** **`attn_scale_base = 1.0`**：不再搜索 base 参数，确保原始上下文窗口内的温度严格等于 1.0

* **单参数搜索**：只搜索 `attn_scale_coef` 参数

* **效果优先的搜索策略**：优先保证找到最优参数，使用密集采样和精细搜索

* **提高搜索精度**：使用更密集的采样点，确保不遗漏最优参数

## Impact

* Affected files:

  * `search_attn_scale_params.py`：恢复单参数搜索，优化搜索策略

  * `search_params.sh`：更新参数配置

* Affected systems:

  * 参数搜索系统

  * RoPE 温度缩放系统

## ADDED Requirements

### Requirement: 严格的原始上下文窗口约束

系统应确保在原始上下文窗口内，温度严格等于 1.0。

#### Scenario: 原始上下文窗口约束

* **WHEN** token 位置在原始上下文窗口内（`t_eff <= original_max_position_embeddings`）

* **THEN** 温度 `t_base = 1.0`（严格等于）

* **AND** `attn_scale_base` 应固定为 1.0

* **AND** 不需要搜索 `attn_scale_base` 参数

#### Scenario: 扩展上下文窗口

* **WHEN** token 位置超出原始上下文窗口（`t_eff > original_max_position_embeddings`）

* **THEN** 温度 `t_base = 1.0 + attn_scale_coef * log(s)`

* **AND** 温度会随着位置增加而增加

### Requirement: 效果优先的搜索策略

系统应采用效果优先的搜索策略，确保找到最优参数。

#### Scenario: 密集采样

* **WHEN** 执行网格搜索时

* **THEN** 应使用密集的采样点（至少 50 个）

* **AND** 确保覆盖整个参数范围

* **AND** 不遗漏可能的最优参数

#### Scenario: 精细搜索

* **WHEN** 执行自适应搜索时

* **THEN** 应使用更多的阶段（至少 4-5 个阶段）

* **AND** 每个阶段应有足够的采样点

* **AND** 最终收敛到最优参数

#### Scenario: 贝叶斯优化

* **WHEN** 执行贝叶斯优化时

* **THEN** 应使用足够的迭代次数（至少 50 次）

* **AND** 初始采样应覆盖整个参数范围

* **AND** 优化过程应充分探索参数空间

### Requirement: 单参数搜索

系统应只搜索 `attn_scale_coef` 参数。

#### Scenario: 参数范围

* **WHEN** 执行参数搜索时

* **THEN** 只搜索 `attn_scale_coef` 参数

* **AND** `attn_scale_base` 固定为 1.0

* **AND** `attn_scale_coef` 范围应为 `[0.05, 0.3]`

* **AND** 参数精度应为三位小数

## MODIFIED Requirements

### Requirement: 参数搜索接口

系统应恢复单参数搜索接口。

**修改前**：

* 同时搜索 `attn_scale_base` 和 `attn_scale_coef`

**修改后**：

* 固定 `attn_scale_base = 1.0`

* 只搜索 `attn_scale_coef`

* 移除 `--attn-scale-base-*` 参数

### Requirement: 搜索空间生成

系统应生成单参数的搜索空间。

**修改前**：

* 生成 `(base, coef)` 参数组合

**修改后**：

* 只生成 `attn_scale_coef` 的参数值

* 使用密集采样（至少 50 个点）

## REMOVED Requirements

### Requirement: 双参数搜索

**Reason**：约束条件要求 `attn_scale_base` 必须固定为 1.0，不需要搜索
**Migration**：恢复单参数搜索，固定 `attn_scale_base = 1.0`

### Requirement: 约束优化

**Reason**：单参数搜索不需要约束优化
**Migration**：使用标准的一维优化方法

## Technical Details

### 温度计算公式

```python
s = torch.clamp(t_eff * self._inv_original_max_pos, min=1.0)
t_base = 1.0 + attn_scale_coef * torch.log(s)  # attn_scale_base 固定为 1.0
```

### 约束条件分析

1. **原始上下文窗口内**（`t_eff <= original_max_position_embeddings`）：

   * `s = 1.0`

   * `log(s) = 0`

   * `t_base = 1.0`（严格等于）

2. **扩展上下文窗口**（`t_eff > original_max_position_embeddings`）：

   * `s > 1.0`

   * `log(s) > 0`

   * `t_base = 1.0 + attn_scale_coef * log(s) > 1.0`

   * 温度随位置增加而增加

### 推荐参数范围

* `attn_scale_coef`: `[0.05, 0.3]`

  * 下限 0.05：确保有足够的温度增长

  * 上限 0.3：避免温度增长过快

  * 采样点：至少 50 个（网格搜索）

### 效果优先的搜索策略

1. **网格搜索**：

   * 采样点：50-100 个

   * 优点：保证不遗漏最优参数

   * 缺点：计算量大

2. **自适应搜索**：

   * 阶段数：4-5 个

   * 初始采样：20-30 个点

   * 优点：平衡效果和效率

   * 缺点：可能陷入局部最优

3. **贝叶斯优化**：

   * 迭代次数：50-100 次

   * 初始采样：10-20 个点

   * 优点：高效找到全局最优

   * 缺点：需要足够的迭代次数

