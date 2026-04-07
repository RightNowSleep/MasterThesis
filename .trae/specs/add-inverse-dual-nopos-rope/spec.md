# 低频维度无位置编码 RoPE 类 Spec

## Why
当前 `LlamaInverseDualRoPEEmbedding` 和 `LlamaInverseDualRoPEScaledEmbedding` 的低频维度（i >= i_star）采用循环位置编码（t % L_0），这可能导致低频分量在长文本中产生不必要的周期性干扰。需要提供一种新的方案：**低频维度完全不施加位置编码**，使其成为位置无关的语义空间，从而更好地捕获全局语义信息。

## What Changes
- **新增类**: `LlamaInverseDualNoPosRoPEEmbedding` — 基础版，高频用全局单调位置 t，低频无位置编码
- **新增类**: `LlamaInverseDualNoPosRoPEScaledEmbedding` — 扩展版，继承基础版并附加 BiFactor 缩放
- **修改配置**: 在 `LlamaConfig._rope_scaling_validation()` 中注册两种新类型
- **修改模型构建**: 在 `LlamaAttention._init_rope()` 中添加新类型的实例化逻辑
- **保持向后兼容**: 不影响现有所有 rope_scaling 类型

## Impact
- Affected specs: 无（全新功能）
- Affected code:
  - `models/pe_llama.py`: 新增两个类定义
  - `models/configuration_llama.py`: 注册新类型及参数验证
  - `models/modeling_llama.py`: _init_rope() 分支扩展

## ADDED Requirements

### Requirement: 低频无位置编码基础类
系统 SHALL 提供 `LlamaInverseDualNoPosRoPEEmbedding` 类，实现以下行为：

#### 设计原理
```
inv_freq: 完整, size = dim // 2
- 高频维度 (i < i_star): position = t (全局单调递增)
- 低频维度 (i >= i_star): position = 0 (恒为零，即 cos=1, sin=0)
```

#### Scenario: 正常初始化与缓存
- **WHEN** 使用默认参数初始化 `LlamaInverseDualNoPosRoPEEmbedding`
- **THEN** 应正确计算 i_star 分割点，高频部分使用全局位置索引，低频部分频率乘以 0（即 cos=1, sin=0）

#### Scenario: Forward 推理
- **WHEN** 调用 forward(x) 且 seq_len 未超过缓存
- **THEN** 返回缓存的 cos/sin 张量，其中低频维度的 cos 全为 1、sin 全为 0

### Requirement: 低频无位置编码缩放类
系统 SHALL 提供 `LlamaInverseDualNoPosRoPEScaledEmbedding` 类：

#### 设计原理
继承 `LlamaInverseDualNoPosRoPEEmbedding`，并在其基础上叠加 BiFactor 缩放函数 s(t) = global(t) × local(t)，仅应用于高频维度部分。

#### Scenario: 缩放因子计算
- **WHEN** seq_len > L_0 时调用 `_compute_attn_scale`
- **THEN** 返回的 attn_scale 对 t < L_0 为 1.0，对 t >= L_0 为 `(1 + α·ln(k+1)) · (1 + β·(1-r)^(1/γ))`

### Requirement: 配置类型注册
系统 SHALL 在 `configuration_llama.py` 中注册以下新类型：
- `"inverse-dual-nopos-rope"` → 对应 `LlamaInverseDualNoPosRoPEEmbedding`
- `"inverse-dual-nopos-rope-scaled"` → 对应 `LlamaInverseDualNoPosRoPEScaledEmbedding`

#### Scenario: 配置验证通过
- **WHEN** config 中设置 `rope_scaling = {"type": "inverse-dual-nopos-rope-scaled", "factor": 4.0}`
- **THEN** 验证通过，且 alpha/beta/gamma 可选参数有默认值

### Requirement: 模型构建集成
系统 SHALL 在 `modeling_llama.py` 的 `LlamaAttention._init_rope()` 方法中支持新类型的选择和实例化。

#### Scenario: 模型正确实例化
- **WHEN** config.rope_scaling.type == "inverse-dual-nopos-rope"
- **THEN** self.rotary_emb 被赋值为 `LlamaInverseDualNoPosRoPEEmbedding` 实例

#### Scenario: 带缩放的模型实例化
- **WHEN** config.rope_scaling.type == "inverse-dual-nopos-rope-scaled"
- **THEN** self.rotary_emb 被赋值为 `LlamaInverseDualNoPosRoPEScaledEmbedding` 实例，并传入 alpha/beta/gamma 参数

## MODIFIED Requirements
无（纯增量功能）

## REMOVED Requirements
无
