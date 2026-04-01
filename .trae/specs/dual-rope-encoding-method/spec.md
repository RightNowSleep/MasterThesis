# 双重RoPE编码方法 Spec

## Why
用户希望创建一种全新的双重RoPE编码方法，通过将维度分成两部分，分别使用不同的位置索引计算方式，以更好地处理长序列位置编码。

## What Changes
- 新增 `LlamaDualRoPEEmbedding` 类：实现双重RoPE编码
- 新增 `LlamaDualRoPEScaledEmbedding` 类：带注意力温度缩放的双重RoPE
- 第一重RoPE：前半部分维度，位置索引 = t % S（模运算，循环使用）
- 第二重RoPE：后半部分维度，位置索引 = t // S（整除运算，扩展范围）

## Impact
- Affected code: `models/pe_llama.py`

## ADDED Requirements

### Requirement: 双重RoPE位置编码
系统应当提供一种双重RoPE编码方法，将维度分成两部分，分别使用不同的位置索引计算方式。

#### Scenario: 双重RoPE编码计算
- **WHEN** 给定位置 t 和缩放因子 S
- **THEN** 
  - 前半部分维度使用位置索引 `t % S`（模运算）
  - 后半部分维度使用位置索引 `t // S`（整除运算）

### Requirement: 缩放因子S的计算
系统应当根据序列长度动态计算缩放因子S。

#### Scenario: 动态缩放因子
- **WHEN** 序列长度超过原始上下文窗口
- **THEN** S = max(1, seq_len / original_max_position_embeddings)

### Requirement: 注意力温度缩放版本
系统应当提供带注意力温度缩放的双重RoPE版本。

#### Scenario: 注意力温度缩放
- **WHEN** 使用 `LlamaDualRoPEScaledEmbedding`
- **THEN** 在 cos/sin 值上应用温度缩放因子

## Design Details

### 数学公式

对于位置 t 和缩放因子 S：

```
# 第一重RoPE（前半部分维度，dim // 2）
pos_1 = t % S  # 模运算，范围 [0, S-1]

# 第二重RoPE（后半部分维度，dim // 2）
pos_2 = t // S  # 整除运算，范围 [0, seq_len // S]
```

### 设计思想

1. **第一重RoPE (t % S)**：
   - 使用模运算，位置索引在 [0, S-1] 范围内循环
   - 适合捕获局部位置信息
   - 当 S = original_max_position_embeddings 时，在原始窗口内位置编码不变

2. **第二重RoPE (t // S)**：
   - 使用整除运算，位置索引随位置增长
   - 适合捕获全局位置信息
   - 可以表示更大的位置范围

### 类结构

```python
class LlamaDualRoPEEmbedding(nn.Module):
    """
    Dual RoPE Embedding.
    
    Splits the dimension into two parts:
    - First half: position index = t % S (modulo operation)
    - Second half: position index = t // S (integer division)
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device=None,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 2048,
        dynamic: bool = False,
    ):
        ...
    
    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        # Compute S
        S = max(1, seq_len / self.original_max_position_embeddings)
        
        # First half: t % S
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        pos_1 = t % S
        
        # Second half: t // S
        pos_2 = t // S
        
        # Compute frequencies for each half
        dim_half = self.dim // 4  # Each half uses dim // 2, split into cos/sin pairs
        inv_freq_1 = 1.0 / (self.base ** (torch.arange(0, dim_half * 2, 2).float() / (dim_half * 2)))
        inv_freq_2 = 1.0 / (self.base ** (torch.arange(0, dim_half * 2, 2).float() / (dim_half * 2)))
        
        freqs_1 = pos_1[:, None] * inv_freq_1[None, :]
        freqs_2 = pos_2[:, None] * inv_freq_2[None, :]
        
        # Concatenate: [cos1, sin1, cos2, sin2] -> [cos1, cos2, sin1, sin2]
        cos = torch.cat([freqs_1.cos(), freqs_2.cos()], dim=-1)
        sin = torch.cat([freqs_1.sin(), freqs_2.sin()], dim=-1)
        ...
```
