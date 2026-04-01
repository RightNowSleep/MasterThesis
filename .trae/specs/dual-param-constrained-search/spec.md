# 双参数约束搜索规范

## Why
单参数搜索（固定 `attn_scale_base=1.0`）虽然简化了问题，但限制了模型在特定长度下的性能优化空间。双参数搜索可以找到更优的组合，但需要添加约束条件：在原始上下文窗口内，温度不应超过 1.0，以保持原始模型的设计意图。

## What Changes
- **恢复双参数搜索**：同时搜索 `attn_scale_base` 和 `attn_scale_coef`
- **添加约束条件**：`attn_scale_base <= 1.0`（确保在原始上下文窗口内温度不超过 1）
- **优化搜索空间**：基于约束条件设计合理的参数范围
- **改进搜索策略**：使用约束优化方法，提高搜索效率

## Impact
- Affected files:
  - `search_attn_scale_params.py`：恢复双参数搜索逻辑，添加约束条件
  - `search_params.sh`：更新参数配置
  - `models/pe_llama.py`：验证温度计算逻辑
- Affected systems:
  - 参数搜索系统
  - RoPE 温度缩放系统

## ADDED Requirements

### Requirement: 双参数同步搜索
系统应支持同时搜索 `attn_scale_base` 和 `attn_scale_coef` 两个参数。

#### Scenario: 参数搜索空间
- **WHEN** 执行参数搜索时
- **THEN** 应同时搜索两个参数
- **AND** `attn_scale_base` 范围应为 `[0.5, 1.0]`（约束条件）
- **AND** `attn_scale_coef` 范围应为 `[0.05, 0.3]`
- **AND** 参数精度应为三位小数

### Requirement: 原始上下文窗口约束
系统应确保在原始上下文窗口内，温度不超过 1.0。

#### Scenario: 温度约束验证
- **WHEN** 计算温度 `t_base = attn_scale_base + attn_scale_coef * log(s)` 时
- **AND** 在原始上下文窗口内（`s = 1.0`，即 `log(s) = 0`）
- **THEN** `t_base = attn_scale_base`
- **AND** 应满足约束 `attn_scale_base <= 1.0`

#### Scenario: 扩展上下文窗口
- **WHEN** 在扩展的上下文窗口内（`s > 1.0`，即 `log(s) > 0`）
- **THEN** `t_base = attn_scale_base + attn_scale_coef * log(s) > attn_scale_base`
- **AND** 温度会随着上下文长度增加而增加

### Requirement: 约束优化搜索
系统应使用约束优化方法进行参数搜索。

#### Scenario: 网格搜索约束
- **WHEN** 使用网格搜索时
- **THEN** 只生成满足约束的参数组合
- **AND** `attn_scale_base` 只在 `[0.5, 1.0]` 范围内采样

#### Scenario: 贝叶斯优化约束
- **WHEN** 使用贝叶斯优化时
- **THEN** 应使用约束优化方法（如 L-BFGS-B with bounds）
- **AND** 边界条件应包含 `attn_scale_base <= 1.0`

#### Scenario: 自适应搜索约束
- **WHEN** 使用自适应搜索时
- **THEN** 应在约束范围内调整搜索中心
- **AND** 搜索半径不应超出约束边界

## MODIFIED Requirements

### Requirement: 参数搜索接口
系统应恢复双参数搜索接口。

**修改前**：
- 只搜索 `attn_scale_coef`，固定 `attn_scale_base=1.0`

**修改后**：
- 同时搜索 `attn_scale_base` 和 `attn_scale_coef`
- 添加 `--attn-scale-base-min`、`--attn-scale-base-max`、`--attn-scale-base-steps` 参数
- 默认值：`base` 范围 `[0.5, 1.0]`，`coef` 范围 `[0.05, 0.3]`

### Requirement: 搜索空间生成
系统应生成满足约束的参数组合。

**修改前**：
- 只生成 `attn_scale_coef` 的参数值

**修改后**：
- 生成 `(attn_scale_base, attn_scale_coef)` 的参数组合
- 确保 `attn_scale_base <= 1.0`
- 使用三位小数精度

## REMOVED Requirements

### Requirement: 单参数搜索
**Reason**：单参数搜索限制了优化空间，双参数搜索可以找到更优的组合
**Migration**：恢复双参数搜索，添加约束条件

## Technical Details

### 温度计算公式
```python
s = torch.clamp(t_eff * self._inv_original_max_pos, min=1.0)
t_base = self.attn_scale_base + self.attn_scale_coef * torch.log(s)
```

### 约束条件分析
1. **原始上下文窗口内**（`t_eff <= original_max_position_embeddings`）：
   - `s = 1.0`
   - `log(s) = 0`
   - `t_base = attn_scale_base`
   - 约束：`attn_scale_base <= 1.0`

2. **扩展上下文窗口**（`t_eff > original_max_position_embeddings`）：
   - `s > 1.0`
   - `log(s) > 0`
   - `t_base = attn_scale_base + attn_scale_coef * log(s) > attn_scale_base`
   - 温度会随长度增加而增加

### 推荐参数范围
- `attn_scale_base`: `[0.5, 1.0]`
  - 下限 0.5：避免温度过低
  - 上限 1.0：约束条件
- `attn_scale_coef`: `[0.05, 0.3]`
  - 下限 0.05：确保有足够的温度增长
  - 上限 0.3：避免温度增长过快
