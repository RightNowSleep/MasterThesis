# Inverse-Dual-RoPE 实现规范

## Why

现有的 Dual-RoPE 方法将位置索引分为两部分：高频维度使用取余操作(t % S)捕获局部位置，低频维度使用除法操作(t // S)捕获全局位置。为了探索相反的策略效果，需要实现一个反转版本：高频区域采用除法，低频区域采用取余，并且分母只与原始上下文长度有关，与缩放因子无关。

## What Changes

- 在 `models/pe_llama.py` 中新增两个类：
  - `LlamaInverseDualRoPEEmbedding`: Inverse-Dual-RoPE 基础实现
  - `LlamaInverseDualRoPEScaledEmbedding`: 带 attention temperature scaling 的版本
- 在 `models/modeling_llama.py` 中添加对新类型的路由支持
- 在 `models/configuration_llama.py` 的 valid_types 列表中添加新类型
- 在 `models/model_loader.py` 的 _ROPE_TYPES_WITH_DYNAMIC_FLAG 中添加新类型
- 在 `continued_pretrain.py` 和 `continued_pretrain.sh` 中实现渐进式长度训练(Progressive Length Training)功能

## Impact

- Affected specs: 无
- Affected code:
  - models/pe_llama.py (新增 ~150 行)
  - models/modeling_llama.py (修改 ~20 行)
  - models/configuration_llama.py (修改 ~5 行)
  - models/model_loader.py (修改 ~3 行)
  - continued_pretrain.py (修改 ~80 行)
  - continued_pretrain.sh (修改 ~15 行)

## ADDED Requirements

### Requirement: Inverse-Dual-RoPE 核心实现

系统 SHALL 提供 Inverse-Dual-RoPE 的位置编码实现，具有以下特性：

#### 算法描述

基于 i_star 临界维度的双位置编码反转策略：
- **inv_freq**: 完整频率向量，size = dim // 2
- **i_star 计算**: 与 Dual-RoPE 相同，r_i = L_0 * θ_i / (2π)，i_star = 第一个 r_i < 1 的索引
- **高频区域 (i < i_start)**: position_index = t // L_0 （除法，L_0 为原始上下文长度）
- **低频区域 (i >= i_start)**: position_index = t % L_0 （取余，L_0 为原始上下文长度）

**关键差异**:
1. 高频/低频的操作方式与 Dual-RoPE 相反
2. 分母使用原始上下文长度 L_0 而非 S (S = max(scaling_factor, L_0))

#### Scenario: 成功实例化并计算位置编码

- **WHEN** 使用 type="inverse-dual-rope", factor=4.0, max_length=8192, original_length=2048 初始化
- **THEN** 系统 SHALL 正确计算 i_star，并在 forward 时返回正确的 cos/sin 值
- **AND** 高频维度 SHALL 使用 t // 2048 计算位置
- **AND** 低频维度 SHALL 使用 t % 2048 计算位置

### Requirement: Inverse-Dual-RoPE-Scaled 版本

系统 SHALL 提供 Inverse-Dual-RoPE 的 scaled 版本，继承基础实现并添加 attention temperature scaling。

#### Scaling 公式

使用熵基公式: mscale = max(1.0, log_{L_0}(t)) = max(1.0, ln(t) / ln(L_0))

#### Scenario: Scaled 版本正确应用温度缩放

- **WHEN** 使用 type="inverse-dual-rope-scaled" 初始化
- **THEN** 系统 SHALL 在计算 cos/sin 后乘以 mscale 因子
- **AND** 当 t <= L_0 时 mscale = 1.0，当 t > L_0 时 mscale > 1.0

### Requirement: 渐进式长度训练 (Progressive Length Training)

系统 SHALL 支持在继续预训练时使用渐进式长度策略，通过 args 控制。

#### 算法描述

当启用渐进式训练时(max_length > original_max_position_embeddings):
1. 生成长度序列: [original_length, original_length*2, original_length*4, ..., max_length]
2. 每个阶段使用固定长度的数据进行训练
3. 阶段间平滑过渡，避免突变

#### Scenario: 启用渐进式训练

- **WHEN** 设置 --progressive-length=True, --max-length=16384, original_length=2048
- **THEN** 系统 SHALL 自动生成长度序列 [2048, 4096, 8192, 16384]
- **AND** 按顺序在每个长度上进行训练
- **AND** 每个 length 阶段的训练步数按比例分配或可配置

## MODIFIED Requirements

### Requirement: 配置验证扩展

configuration_llama.py 中的 `_rope_scaling_validation()` 方法 SHALL 扩展以支持新类型:

在 valid_types 列表中添加:
- "inverse-dual-rope"
- "inverse-dual-rope-scaled"

### Requirement: 模型加载器类型注册

model_loader.py 中的 `_ROPE_TYPES_WITH_DYNAMIC_FLAG` 集合 SHALL 包含新类型:
- "inverse-dual-rope"
- "inverse-dual-rope-scaled"

### Requirement: 模型构建路由

modeling_llama.py 中的 `_init_rope()` 方法 SHALL 添加路由分支:
- "inverse-dual-rope" → LlamaInverseDualRoPEEmbedding
- "inverse-dual-rope-scaled" → LlamaInverseDualRoPEScaledEmbedding

## REMOVED Requirements

无
