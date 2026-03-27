import math
import numpy as np
import torch
from torch import nn


class LlamaEntropyStableRotaryEmbedding(nn.Module):
    """
    Entropy-Stable RoPE - 基于熵稳定的注意力缩放

    Parameters
    ----------
    dim : int
        注意力头维度
    max_position_embeddings : int
        最大位置编码长度
    base : int
        RoPE基数，默认10000
    device : torch.device
        计算设备
    scaling_factor : float
        静态扩展比例S
    original_max_position_embeddings : int
        原始训练长度L_0
    dynamic : bool
        是否动态计算缩放
    alpha : float
        基础缩放系数（默认0.1，与YaRN相同）
    beta : float
        熵补偿强度（默认1.0）
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
        alpha: float = 0.1,
        beta: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scaling_factor = scaling_factor
        self.dynamic = dynamic
        self.alpha = alpha
        self.beta = beta

        # 预计算基础参数
        self._precompute_params(device)

        # 预计算熵补偿因子
        self._precompute_compensation_factors()

        if not dynamic:
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings,
                device=device,
                dtype=torch.get_default_dtype(),
            )

    def _precompute_params(self, device):
        """预计算基础参数"""
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _precompute_compensation_factors(self):
        """预计算熵补偿因子"""
        L0 = self.original_max_position_embeddings
        max_pos = 16384  # 预计算到16k

        factors = []
        for pos in range(max_pos):
            if pos < L0:
                # 训练长度内：不补偿
                factor = 1.0
            else:
                # 训练长度外：对数增长的补偿
                rel_pos = pos - L0
                rel_ratio = rel_pos / L0

                # 相对熵（对数增长模型）
                rel_entropy = 1.0 + 0.5 * math.log(1 + rel_ratio) / math.log(2)

                # 补偿因子
                factor = 1.0 + self.beta * (rel_entropy - 1.0)

                # 限制最大补偿
                factor = min(factor, 2.5)

            factors.append(factor)

        self.compensation_factors = torch.tensor(factors, dtype=torch.float32)

    def _compute_compensation_factor(self, position: int) -> float:
        """计算熵补偿因子"""
        if position < len(self.compensation_factors):
            return self.compensation_factors[position].item()
        else:
            # 超出预计算范围，使用最大值
            return 2.5

    def _compute_attention_scaling(self, position: int, S: float) -> float:
        """
        计算注意力缩放系数

        t(position, S) = 1 + α · ln(S) · compensation_factor(position)
        """
        if S <= 1.0:
            return 1.0

        # 熵补偿因子
        compensation_factor = self._compute_compensation_factor(position)

        # 综合缩放
        t = 1.0 + self.alpha * math.log(S) * compensation_factor

        return t

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """设置cos/sin缓存"""
        self.max_seq_len_cached = seq_len
        inv_freq = self.inv_freq.to(device=device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        # 标准RoPE
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # 应用熵稳定缩放
        S = self.scaling_factor
        scaling_factors = torch.tensor(
            [self._compute_attention_scaling(int(pos), S) for pos in range(seq_len)],
            device=device,
            dtype=dtype,
        )

        self.register_buffer(
            "cos_cached",
            emb.cos().to(dtype) * scaling_factors[:, None],
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            emb.sin().to(dtype) * scaling_factors[:, None],
            persistent=False,
        )

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """
        前向传播

        Parameters
        ----------
        x : torch.Tensor
            输入张量 [batch, num_heads, seq_len, head_dim]
        seq_len : int
            序列长度（动态模式下使用）

        Returns
        -------
        cos, sin : torch.Tensor
            旋转编码的余弦和正弦分量，已应用熵稳定缩放
        """
        device, dtype = x.device, x.dtype
        if seq_len is None:
            seq_len = x.shape[2]

        if self.dynamic:
            # 动态模式：实时计算缩放
            S = max(1.0, seq_len / self.original_max_position_embeddings)
            inv_freq = self.inv_freq.to(device=device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)

            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            # 动态计算熵稳定缩放
            scaling_factors = torch.tensor(
                [
                    self._compute_attention_scaling(int(pos), S)
                    for pos in range(seq_len)
                ],
                device=device,
                dtype=dtype,
            )

            return (
                emb.cos().to(dtype=dtype) * scaling_factors[:, None],
                emb.sin().to(dtype=dtype) * scaling_factors[:, None],
            )
        else:
            # 静态模式：使用缓存
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
            return (
                self.cos_cached[:seq_len].to(dtype=dtype),
                self.sin_cached[:seq_len].to(dtype=dtype),
            )
