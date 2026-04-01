# 修改温度缩放公式规范

## Why
当前的温度缩放公式在原始上下文窗口内的温度不严格等于 1.0，需要修改为新的公式，确保在原始上下文窗口内温度严格等于 1.0，并且在扩展上下文窗口内温度按对数比例增长。

## What Changes
- **修改温度缩放公式**：从 `t_base = attn_scale_base + attn_scale_coef * log(s)` 改为 `mscale_i(t) = 1 + α * max(0, (ln(max(1, floor(t/b_i))) - ln(L_0)) / ln(L_0))`
- **注释旧实现**：保留旧代码作为注释，便于对比和回退
- **简化参数**：移除 `attn_scale_base` 参数，只保留 `attn_scale_coef`（对应 α）
- **更新参数搜索**：调整参数搜索范围和默认值

## Impact
- Affected files:
  - `models/pe_llama.py`：修改 `_compute_attn_scale` 方法
  - `search_attn_scale_params.py`：更新参数搜索配置
  - `search_params.sh`：更新参数配置

## ADDED Requirements

### Requirement: 新的温度缩放公式
系统应使用新的温度缩放公式，确保在原始上下文窗口内温度严格等于 1.0。

#### Scenario: 原始上下文窗口内
- **WHEN** `floor(t / b_i) <= L_0`
- **THEN** `mscale_i(t) = 1.0`（严格等于）
- **AND** 温度不随位置变化

#### Scenario: 扩展上下文窗口
- **WHEN** `floor(t / b_i) > L_0`
- **THEN** `mscale_i(t) = 1 + α * (ln(floor(t/b_i)) - ln(L_0)) / ln(L_0)`
- **AND** 温度随位置对数增长

### Requirement: 参数简化
系统应简化温度缩放参数，只保留一个可调参数。

#### Scenario: 参数配置
- **WHEN** 配置温度缩放参数时
- **THEN** 只需要配置 `attn_scale_coef`（对应 α）
- **AND** 移除 `attn_scale_base` 参数
- **AND** 默认值 `attn_scale_coef = 0.1`

## MODIFIED Requirements

### Requirement: 温度缩放计算
系统应使用新的公式计算温度缩放。

**修改前**：
```python
s = torch.clamp(t_eff * self._inv_original_max_pos, min=1.0)
t_base = self.attn_scale_base + self.attn_scale_coef * torch.log(s)
```

**修改后**：
```python
# 新公式: mscale_i(t) = 1 + α * max(0, (ln(max(1, floor(t/b_i))) - ln(L_0)) / ln(L_0))
# 其中:
#   t: token 位置 (t_eff)
#   b_i: block size
#   L_0: 原始上下文窗口大小 (original_max_position_embeddings)
#   α: 可调参数 (attn_scale_coef)

# 计算 floor(t / b_i)
t_clipped = torch.clamp(t_eff, min=1.0)

# 计算 ln(max(1, floor(t/b_i)))
log_t = torch.log(t_clipped)

# 计算 ln(L_0)
log_L0 = math.log(self.original_max_position_embeddings)

# 计算 (ln(floor(t/b_i)) - ln(L_0)) / ln(L_0)
normalized_log = (log_t - log_L0) / log_L0

# 计算 max(0, ...)
clipped_log = torch.clamp(normalized_log, min=0.0)

# 计算 1 + α * ...
mscale = 1.0 + self.attn_scale_coef * clipped_log
```

### Requirement: 参数搜索范围
系统应调整参数搜索范围。

**修改前**：
- `attn_scale_coef` 范围：`[0.05, 0.3]`

**修改后**：
- `attn_scale_coef` 范围：`[0.05, 0.2]`（更合理的范围）
- 默认值：`0.1`

## REMOVED Requirements

### Requirement: attn_scale_base 参数
**Reason**：新公式不需要 `attn_scale_base` 参数，温度在原始上下文窗口内严格等于 1.0
**Migration**：移除 `attn_scale_base` 参数，固定为 1.0

## Technical Details

### 公式推导

新公式的数学表达：
$$\text{mscale}_i(t) = 1 + \alpha \cdot \max\!\left(0,\ \frac{\ln\max\!\left(1,\ \left\lfloor t / b_i \right\rfloor\right) - \ln L_0}{\ln L_0}\right)$$

关键特性：
1. **原始上下文窗口内**（`floor(t/b_i) <= L_0`）：
   - `ln(floor(t/b_i)) <= ln(L_0)`
   - `normalized_log <= 0`
   - `clipped_log = 0`
   - `mscale_i(t) = 1.0`

2. **扩展上下文窗口**（`floor(t/b_i) > L_0`）：
   - `ln(floor(t/b_i)) > ln(L_0)`
   - `normalized_log > 0`
   - `clipped_log = normalized_log`
   - `mscale_i(t) = 1 + α * (ln(floor(t/b_i)) - ln(L_0)) / ln(L_0)`

3. **对数增长**：
   - 温度随位置的对数增长
   - 增长速度由 α 控制

### 参数推荐范围
- `attn_scale_coef` (α): `[0.05, 0.2]`
  - 下限 0.05：确保有足够的温度增长
  - 上限 0.2：避免温度增长过快
  - 默认值 0.1：平衡的温度增长
