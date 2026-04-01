# Dual-RoPE 性能优化 Spec

## Why
dual-rope 和 dual-rope-scaled 的当前实现存在多个性能瓶颈：
1. 重复的张量切片操作（`inv_freq[:i_star]` 和 `inv_freq[i_star:]`）
2. 多次中间张量创建和内存拷贝
3. 在 scaled 版本中重复计算位置信息
4. attention scale 计算可以进一步优化

这些优化将减少内存分配、降低计算开销，提升训练和推理速度。

## What Changes
- 预计算并缓存分割后的 inv_freq 部分，避免每次 forward 重复切片
- 合并张量操作以减少中间内存分配
- 优化 LlamaDualRoPEScaledEmbedding 避免重复计算位置信息
- 使用更高效的张量操作（如 `torch.fmod`、`torch.div`）
- 优化 attention scale 计算逻辑

## Impact
- Affected specs: RoPE 位置编码模块
- Affected code:
  - `models/pe_llama.py` 中的 `LlamaDualRoPEEmbedding` 类
  - `models/pe_llama.py` 中的 `LlamaDualRoPEScaledEmbedding` 类

## ADDED Requirements

### Requirement: 预计算分割后的 inv_freq
系统应当在初始化时预计算并缓存分割后的 inv_freq 部分（`inv_freq_1` 和 `inv_freq_2`），避免在每次 forward 调用时重复执行切片操作。

#### Scenario: 静态模式初始化
- **WHEN** 使用静态模式（dynamic=False）初始化 LlamaDualRoPEEmbedding
- **THEN** 系统应创建 `inv_freq_1` 和 `inv_freq_2` 两个缓冲区，分别存储 `inv_freq[:i_star]` 和 `inv_freq[i_star:]`

#### Scenario: 动态模式初始化
- **WHEN** 使用动态模式（dynamic=True）初始化 LlamaDualRoPEEmbedding
- **THEN** 系统应创建 `inv_freq_1` 和 `inv_freq_2` 两个缓冲区，以便在 forward 时直接使用

### Requirement: 优化张量操作以减少内存分配
系统应当合并多个张量操作，减少中间张量的创建和内存拷贝。

#### Scenario: 位置编码计算
- **WHEN** 计算 cos/sin 缓存时
- **THEN** 系统应使用预计算的 inv_freq 部分，避免重复切片
- **THEN** 系统应尽量减少 `torch.cat` 和 `repeat` 操作的次数

### Requirement: 避免 scaled 版本中的重复计算
LlamaDualRoPEScaledEmbedding 应当避免在 `_set_cos_sin_cache` 中重复计算位置信息。

#### Scenario: 缓存设置
- **WHEN** 调用 `_set_cos_sin_cache` 方法时
- **THEN** 系统应复用父类计算的位置信息，而不是重新计算

### Requirement: 优化 attention scale 计算
系统应当优化 attention scale 的计算逻辑，减少不必要的张量操作。

#### Scenario: Attention scale 计算
- **WHEN** 计算 attention scale 时
- **THEN** 系统应使用向量化操作，避免循环
- **THEN** 系统应尽量减少中间张量的创建

## MODIFIED Requirements

### Requirement: LlamaDualRoPEEmbedding 类优化
原有的 LlamaDualRoPEEmbedding 类需要添加以下优化：

**新增属性：**
- `inv_freq_1`: 预计算的高频部分 inv_freq（前 i_star 个维度）
- `inv_freq_2`: 预计算的低频部分 inv_freq（剩余维度）

**修改方法：**
- `__init__`: 添加 `inv_freq_1` 和 `inv_freq_2` 的初始化
- `_set_cos_sin_cache`: 使用预计算的 `inv_freq_1` 和 `inv_freq_2`
- `forward`: 使用预计算的 `inv_freq_1` 和 `inv_freq_2`

### Requirement: LlamaDualRoPEScaledEmbedding 类优化
原有的 LlamaDualRoPEScaledEmbedding 类需要添加以下优化：

**修改方法：**
- `_set_cos_sin_cache`: 避免重复计算位置信息，复用父类的计算结果
- `forward`: 优化 attention scale 的计算和广播

## REMOVED Requirements
无移除的需求。
