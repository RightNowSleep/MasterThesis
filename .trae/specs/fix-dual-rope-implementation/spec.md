# 修复双重RoPE实现 Spec

## Why

当前的双重RoPE实现有两个问题：

1. 维度划分应该只影响位置索引，而不是改变 inv\_freq 本身
2. 缺少configuration、modeling、model loader的支持，无法正常加载使用

## What Changes

* 修复 `LlamaDualRoPEEmbedding` 的维度划分逻辑：保持 inv\_freq 完整，只将位置索引分成两部分

* 在 `model_loader.py` 中添加 `dual-rope` 和 `dual-rope-scaled` 类型

* 在 `modeling_llama.py` 中添加对应的RoPE初始化逻辑

## Impact

* Affected code:

  * `models/pe_llama.py`

  * `models/model_loader.py`

  * `models/modeling_llama.py`

## ADDED Requirements

### Requirement: 正确的维度划分逻辑

系统应当保持 inv\_freq 完整，只将位置索引分成两部分。

#### Scenario: 维度划分验证

* **WHEN** 给定 head\_dim = 128

* **THEN**

  * inv\_freq 大小 = 64 (dim // 2)，保持完整 `[0, 2, 4, ..., 126]`

  * 前 32 个维度使用位置索引 `t % S`

  * 后 32 个维度使用位置索引 `t // S`

  * 最终 cos/sin 大小 = 128 (dim)

### Requirement: model\_loader 支持

系统应当在 model\_loader.py 中支持 dual-rope 类型。

#### Scenario: 加载 dual-rope 模型

* **WHEN** 用户指定 `--rope-type dual-rope` 或 `--rope-type dual-rope-scaled`

* **THEN** 模型能够正确加载并使用双重RoPE编码

### Requirement: modeling 支持

系统应当在 modeling\_llama.py 中初始化双重RoPE。

#### Scenario: 初始化双重RoPE

* **WHEN** 配置中指定 `rope_scaling.type = "dual-rope"` 或 `"dual-rope-scaled"`

* **THEN** 正确创建 `LlamaDualRoPEEmbedding` 或 `LlamaDualRoPEScaledEmbedding`

## Design Details

### 正确的维度划分逻辑

标准 RoPE:

```python
# dim = head_dim
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
# inv_freq 大小 = dim // 2，例如 [f_0, f_1, f_2, ..., f_{dim//2-1}]
freqs = t[:, None] * inv_freq[None, :]  # [seq_len, dim // 2]
emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
```

Dual RoPE (正确实现):

```python
# inv_freq 保持完整，大小 = dim // 2
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
# inv_freq = [f_0, f_1, f_2, ..., f_{dim//2-1}]

# 将 inv_freq 分成两部分
half_inv_freq = dim // 4  # 每部分的维度

# 第一重: 前 half_inv_freq 个维度使用 t % S
pos_1 = t % S
freqs_1 = pos_1[:, None] * inv_freq[:half_inv_freq][None, :]
# freqs_1 使用 [f_0, f_1, ..., f_{half_inv_freq-1}]

# 第二重: 后 half_inv_freq 个维度使用 t // S
pos_2 = t // S
freqs_2 = pos_2[:, None] * inv_freq[half_inv_freq:][None, :]
# freqs_2 使用 [f_{half_inv_freq}, ..., f_{dim//2-1}]

# 拼接
freqs = torch.cat([freqs_1, freqs_2], dim=-1)  # [seq_len, dim // 2]
cos = freqs.cos().repeat(1, 2)  # [seq_len, dim]
sin = freqs.sin().repeat(1, 2)  # [seq_len, dim]
```

### 示例

假设 dim = 8：

* inv\_freq = \[f\_0, f\_1, f\_2, f\_3] (大小 = 4)

* half\_inv\_freq = 2

对于位置 t 和缩放因子 S：

* pos\_1 = t % S

* pos\_2 = t // S

freqs\_1 = \[pos\_1 \* f\_0, pos\_1 \* f\_1]  # 使用前 2 个频率
freqs\_2 = \[pos\_2 \* f\_2, pos\_2 \* f\_3]  # 使用后 2 个频率

freqs = \[pos\_1 \* f\_0, pos\_1 \* f\_1, pos\_2 \* f\_2, pos\_2 \* f\_3]

这样保持了 inv\_freq 的完整性，只是位置索引分成两部分。

### 需要添加的 RoPE 类型

1. `dual-rope` - 基础双重RoPE
2. `dual-rope-scaled` - 带注意力温度缩放的双重RoPE

