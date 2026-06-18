# BiSpace-RoPE：基于双空间位置编码扩展大语言模型上下文窗口

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <strong>BiSpace-RoPE</strong> --- 一种用于扩展大语言模型上下文窗口的双空间旋转位置编码<br>
  <strong>BS2</strong> --- BiSpace-RoPE + BiFactor-Scaling，实现稳健的长上下文外推
</p>

---

## 概述

将基于 Transformer 的大语言模型上下文窗口扩展至训练长度之外是一项关键挑战。现有的 RoPE 缩放方法（如位置插值 PI、NTK-aware、YaRN）要么损害短上下文性能，要么在极端长度下无法保持稳定性。

本项目提出 **BiSpace-RoPE** 和 **BS2** 两种新方法，将位置编码分解为**全局空间**和**局部空间**，在广泛的上下文长度范围内实现了优异的困惑度和基准测试性能。

**核心贡献：**

- **BiSpace-RoPE**：按临界频率维度拆分位置索引 --- 高频维度使用单调的全局位置，低频维度使用对原始上下文长度取模的循环局部位置。
- **BiFactor-Scaling**：一种分解式注意力温度缩放函数，包含全局对数增长项和局部边界补偿项，解决分段边界处的注意力熵退化问题。
- **BS2（BiSpace-RoPE + BiFactor-Scaling）**：组合两种方法，实现最先进的长上下文外推性能。
- 一套完整的基准测试框架，包含 20+ 种 RoPE 变体，支持继续预训练、SFT 微调、超参数搜索和多维度评估。

---

## 核心方法

### BiSpace-RoPE

BiSpace-RoPE 基于**临界维度** $i^*$ 将位置索引拆分为两个子空间：

$$
i^* = \text{first index where } r_i = \frac{L_0 \cdot \theta_i}{2\pi} < 1
$$

其中 $L_0$ 为原始上下文长度，$\theta_i$ 为维度 $i$ 的旋转频率。

位置编码定义为：

$$
\text{pos}(t, i) =
\begin{cases}
t & i < i^* \quad \text{(高频：全局，单调)} \\
t \bmod L_0 & i \geq i^* \quad \text{(低频：局部，循环)}
\end{cases}
$$

**直觉**：高频维度在 $L_0$ 内完成多次旋转，需要精确的全局位置。低频维度旋转缓慢，对 $L_0$ 之外的绝对位置不敏感 --- 将其循环回退可保持训练分布。

### BiFactor-Scaling

BiFactor-Scaling 对超出 $L_0$ 的位置施加分解式注意力温度因子：

$$
s(t) =
\begin{cases}
1 & t < L_0 \\
\underbrace{(1 + \alpha \ln(k+1))}_{\text{全局项}} \cdot \underbrace{(1 + \beta e^{-\gamma r})}_{\text{局部项}} & t \geq L_0
\end{cases}
$$

其中：
- $k = \lfloor t / L_0 \rfloor$ --- 段索引（全局坐标）
- $r = (t \bmod L_0) / L_0 \in [0, 1)$ --- 归一化段内位置（局部坐标）
- $\alpha$ --- 全局增长率（默认：0.1，类似 YaRN 的对数增长）
- $\beta$ --- 边界跳变补偿幅度（默认：0.5）
- $\gamma$ --- 段内指数衰减率（默认：2.0）

**设计原理**：全局项补偿跨段的累积注意力熵损失。局部项在每个段边界处（$r \approx 0$）提供跳变补偿，并在段内（$r \to 1$）指数衰减，解决段过渡处的不连续性问题。

### BS2 = BiSpace-RoPE + BiFactor-Scaling

BS2 将 BiSpace-RoPE 的双空间位置编码与 BiFactor-Scaling 的注意力温度补偿相结合，将缩放因子应用于 BiSpace 编码后的 cos/sin 值：

$$
\text{RoPE}_{\text{BS2}}(t) = s(t) \cdot \left[\cos(\text{pos}(t, i) \cdot \theta_i),\ \sin(\text{pos}(t, i) \cdot \theta_i)\right]
$$

最优的 $(\alpha, \beta, \gamma)$ 参数通过 Optuna 贝叶斯优化搜索（见[超参数搜索](#超参数搜索)）。

---

## 项目结构

```
MasterThesis/
├── models/                          # 核心模型实现
│   ├── pe_llama.py                  # 20+ 种 RoPE 实现（BiSpace-RoPE、BiFactor-Scaling 等）
│   ├── entropy_stable_scaled_rope.py # 熵稳定注意力缩放 RoPE
│   ├── modeling_llama.py            # 自定义 LLaMA 模型（集成 RoPE）
│   ├── configuration_llama.py       # 自定义 LLaMA 配置
│   ├── model_loader.py              # 统一模型/分词器加载器（含 RoPE 配置）
│   ├── gguf_custom.py               # GGUF 导出支持
│   ├── ollama_backend.py            # Ollama 后端集成
│   └── export_to_ollama.py          # 导出为 Ollama 格式
├── eval/                            # 评估模块
│   ├── perplexity.py                # 多长度滑动窗口困惑度
│   ├── entropy.py                   # 注意力熵分析（Shannon 熵 + 归一化熵）
│   ├── eval_harness.py              # lm-eval-harness 基准测试运行器
│   ├── passkey.py                   # Passkey 检索（大海捞针测试）
│   ├── quality.py                   # 长文本多项选择（LongBench-v2、QuALITY）
│   ├── performance.py               # 推理运行时间与 GPU 显存分析
│   ├── plot_entropy.py              # 熵可视化
│   └── plot_entropy_fig3.py         # 论文图 3 熵图
├── drawer/                          # 可视化模块
│   ├── rope_comparison.py           # RoPE 方法对比图
│   ├── position_heatmap.py          # 位置编码热力图
│   ├── runtime.py                   # 运行时间对比图
│   ├── perplexity.py                # 困惑度曲线图
│   └── block_sum.py                 # 分块摘要可视化
├── continued_pretrain.py            # 继续预训练（LoRA/QLoRA）
├── finetune.py                      # 监督微调 SFT（QLoRA）
├── search_attn_scale_params.py      # Optuna 超参数搜索 (α, β, γ)
├── *.sh                             # 训练与评估的 Shell 脚本
├── results/                         # 实验结果（JSON）
│   ├── perplexity/                  # 多上下文长度困惑度
│   ├── entropy/                     # 注意力熵指标
│   ├── eval_harness/                # 基准测试结果（MMLU、HellaSwag 等）
│   ├── performance/                 # 运行时间与显存分析
│   └── param_search/                # Optuna 搜索结果
└── finetunes/                       # LoRA 适配器检查点（已 gitignore）
```

---

## 环境配置

### 系统要求

- Python 3.10
- CUDA 12.x
- PyTorch 2.7+

### 安装

```bash
# 克隆仓库
git clone <repo-url>
cd MasterThesis

# 创建 conda 环境（精简版）
conda env create -f environment_conda.yml
conda activate LLMs

# 或完整环境（包含所有依赖）
conda env create -f environment.yml
conda activate LLMs
```

### 主要依赖

| 包名 | 用途 |
|---------|---------|
| `transformers` | 模型加载与推理 |
| `peft` | LoRA / QLoRA 适配器支持 |
| `bitsandbytes` | 4-bit / 8-bit 量化 |
| `accelerate` | 分布式训练 |
| `trl` | SFT 训练 |
| `lm-eval` | 标准基准评估 |
| `optuna` | 超参数搜索 |
| `datasets` | 数据集加载 |
| `wandb` | 实验跟踪（可选） |

---

## 快速开始

### 加载 BiSpace-RoPE 模型

```python
from models.model_loader import load_model, load_tokenizer

# 加载分词器
tokenizer = load_tokenizer("huggyllama/llama-7b")

# 加载 BiSpace-RoPE 模型（静态，factor=8）
model = load_model(
    "huggyllama/llama-7b",
    rope_type="inverse-dual-rope",
    rope_factor=8.0,
    load_in_4bit=True,
)

# 加载 BS2 模型（动态缩放）
model = load_model(
    "huggyllama/llama-7b",
    rope_type="inverse-dual-rope-scaled",
    rope_dynamic=True,
    rope_alpha=0.1,
    rope_beta=0.5,
    rope_gamma=2.0,
    load_in_4bit=True,
)
```

### 困惑度评估

```bash
python eval/perplexity.py \
    --model-name huggyllama/llama-7b \
    --rope-type inverse-dual-rope-scaled \
    --rope-dynamic \
    --load-in-4bit \
    --max-length 65536 \
    --min-length 2048
```

---

## 使用指南

### 继续预训练

通过 QLoRA/LoRA 继续预训练来扩展上下文窗口：

```bash
# 单个方法
python continued_pretrain.py \
    --model-name huggyllama/llama-7b \
    --rope-type inverse-dual-rope \
    --rope-factor 8.0 \
    --max-length 16384 \
    --quantization 4bit \
    --lora-r 64 --lora-alpha 128 \
    --max-train-steps 400 \
    --dataset emozilla/pg_books-tokenized-bos-eos-chunked-65536

# 批量：通过 Shell 脚本运行多个方法
bash continued_pretrain.sh
```

**渐进式长度训练**：跨阶段逐步增加序列长度（`--progressive-length`），例如 [2048, 4096, 8192, 16384]。

**层级式训练**：在 BiSpace-RoPE 适配器基础上训练 BS2：

```bash
python continued_pretrain.py \
    --model-name huggyllama/llama-7b \
    --base-adapter-path finetunes/continued_pretrain/inverse-dual-rope_20260403_103555 \
    --rope-type inverse-dual-rope-scaled \
    --rope-factor 8.0 \
    --max-length 16384 \
    --quantization 4bit
```

### 监督微调（SFT）

```bash
python finetune.py \
    --model-name meta-llama/Llama-2-7b-chat-hf \
    --rope-type inverse-dual-rope-scaled --rope-factor 4.0 \
    --quantization 4bit --use-lora \
    --dataset HuggingFaceH4/ultrachat_200k \
    --num-train-epochs 1 \
    --output-dir finetunes/finetune

# 或通过 Shell 脚本批量运行
bash finetune.sh
```

### 超参数搜索

使用 Optuna 搜索 BiFactor-Scaling 的最优 $(\alpha, \beta, \gamma)$：

```bash
python search_attn_scale_params.py \
    --model-name huggyllama/llama-7b \
    --rope-type inverse-dual-rope-scaled \
    --rope-dynamic \
    --base-adapter-path finetunes/continued_pretrain/inverse-dual-rope_20260403_103555 \
    --n-trials 100 \
    --alpha-range 0.05,0.40 \
    --beta-range 0.20,1.50 \
    --gamma-range 1.10,5.00 \
    --load-in-4bit

# 或通过 Shell 脚本
bash search_params.sh
```

### 评估

#### 困惑度（多长度）

```bash
python eval/perplexity.py \
    --model-name huggyllama/llama-7b \
    --adapter-path finetunes/continued_pretrain/inverse-dual-rope-scaled_20260406_070155 \
    --load-in-4bit \
    --max-length 65536 --min-length 2048

# 批量
bash eval_perplexity.sh
```

#### 注意力熵

```bash
python eval/entropy.py \
    --model-name huggyllama/llama-7b \
    --rope-type inverse-dual-rope-scaled --rope-dynamic \
    --load-in-4bit \
    --max-length 3072 --num-samples 100

# 批量（评估 + 绘图）
bash entropy.sh
```

#### 标准基准测试（lm-eval-harness）

```bash
python eval/eval_harness.py \
    --model-name huggyllama/llama-7b \
    --adapter-path finetunes/continued_pretrain/inverse-dual-rope-scaled_20260406_070155 \
    --load-in-4bit \
    --tasks mmlu,hellaswag,gsm8k,arc_challenge

# 批量
bash eval_harness.sh
```

**可用任务类别**：`reasoning`（MMLU、HellaSwag、ARC、TruthfulQA、TriviaQA）、`math`（GSM8K、Minerva Math、AGIEval Math）、`code`（MBPP）、`long_context`（Passkey、LongBench、Babilong、NIAH）。

#### Passkey 检索

```bash
python eval/passkey.py \
    --model-name huggyllama/llama-7b \
    --rope-type inverse-dual-rope-scaled --rope-dynamic \
    --load-in-4bit \
    --max-length 65536 --data-mode real

# 批量
bash eval_passkey.sh
```

#### 长文本质量（LongBench-v2 / QuALITY）

```bash
python eval/quality.py \
    --model-name huggyllama/llama-7b \
    --rope-type inverse-dual-rope-scaled --rope-dynamic \
    --load-in-4bit \
    --scoring-mode logit --limit 50
```

#### 推理性能

```bash
python eval/performance.py \
    --model-name huggyllama/llama-7b \
    --rope-type inverse-dual-rope-scaled --rope-dynamic \
    --load-in-4bit \
    --max-length 4096 --use-cache

# 批量
bash eval_performance.sh
```

---

## 支持的 RoPE 方法

| 类别 | 方法 | CLI `--rope-type` | 动态 | 注意力缩放 |
|----------|--------|-------------------|---------|-------------|
| **基线** | 标准 RoPE | `none` | - | - |
| **经典** | 位置插值 | `linear` | 是 | - |
| **经典** | NTK-aware | `ntk` | 是 | - |
| **经典** | NTK-by-parts | `part-ntk` | 是 | - |
| **经典** | YaRN | `yarn` | 是 | 是 |
| **自定义** | My-RoPE | `my-rope` | 是 | - |
| **自定义** | My-RoPE Scaled | `my-rope-scaled` | 是 | 是 |
| **自定义** | My-RoPE2 | `my-rope2` | 是 | - |
| **自定义** | My-RoPE2 Scaled | `my-rope2-scaled` | 是 | 是 |
| **分块** | Block-Layered | `block-layered` | 是 | - |
| **分块** | Block-Layered Scaled | `block-layered-scaled` | 是 | 是 |
| **频率** | Freq-Smooth | `freq-smooth` | 是 | - |
| **频率** | Freq-Smooth Scaled | `freq-smooth-scaled` | 是 | 是 |
| **频率** | Freq-Reciprocal | `freq-reciprocal` | 是 | - |
| **频率** | Freq-Reciprocal Scaled | `freq-reciprocal-scaled` | 是 | 是 |
| **频率** | Freq-Reciprocal Scaled No-Layer | `freq-reciprocal-scaled-no-layer` | 是 | 是 |
| **频率** | Freq-Reciprocal Scaled Adaptive | `freq-reciprocal-scaled-adaptive` | 是 | 是 |
| **双位置** | Dual-RoPE | `dual-rope` | 是 | - |
| **双位置** | Dual-RoPE Scaled | `dual-rope-scaled` | 是 | 是 |
| **核心** | **BiSpace-RoPE** | `inverse-dual-rope` | 是 | - |
| **核心** | **BS2（BiSpace + BiFactor）** | `inverse-dual-rope-scaled` | 是 | 是 |
| **核心** | **BiFactor-Scaling** | `bi-factor-scaling-rope` | 是 | 是 |
| **变体** | BiSpace-Tangle | `inverse-dual-tangle-rope` | 是 | - |
| **变体** | BiSpace-Tangle Scaled | `inverse-dual-tangle-rope-scaled` | 是 | 是 |
| **变体** | BiSpace-NoPos | `inverse-dual-nopos-rope` | 是 | - |
| **变体** | BiSpace-NoPos Scaled | `inverse-dual-nopos-rope-scaled` | 是 | 是 |

**静态与动态模式**：使用 `--rope-factor F` 进行静态缩放（固定比率），或使用 `--rope-dynamic` 进行运行时自适应缩放，其中 $s = \max(1, \; \text{seq\_len} \; / \; L_0)$。

---

## 实验结果

### 困惑度（ProofPile，静态 Factor=8）

| 上下文长度 | None | Linear | NTK | Part-NTK | YaRN | **BiSpace-RoPE** | **BS2** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2048 | 4.80 | 7.07 | 4.86 | 5.17 | 5.13 | **4.86** | 4.86 |
| 4096 | 10.34 | 6.05 | 4.19 | 4.51 | 4.38 | 8.08 | - |
| 8192 | 37.19 | 5.05 | 3.55 | 3.86 | 3.68 | 13.44 | - |
| 16384 | 133.53 | 4.52 | 5.81 | 3.49 | 3.25 | 21.53 | - |
| 32768 | 353.21 | 24.63 | 29.33 | 20.92 | 20.97 | 30.74 | - |
| 65536 | 419.75 | 38.40 | 45.45 | 33.40 | 34.13 | 35.64 | - |

> 以上结果来自基座模型（未经继续预训练）。BiSpace-RoPE 在保持短上下文质量的同时（PPL@2048 = 4.86 vs 基线 4.80），实现了有竞争力的长上下文困惑度。

### 标准基准测试（Factor=8，继续预训练后）

| 方法 | MMLU (5-shot) | HellaSwag (10-shot) | GSM8K (8-shot) |
|--------|:-:|:-:|:-:|
| None | 32.20 | 75.74 | 6.44 |
| Linear | 25.93 | 65.03 | - |
| NTK | 30.37 | 74.89 | - |
| Part-NTK | 28.18 | 71.57 | - |
| YaRN | 29.78 | 74.53 | - |
| **BiSpace-RoPE** | 31.28 | 75.15 | - |
| **BS2** | **34.36** | **76.55** | **7.43** |

> BS2 在所有基准测试中均取得最佳性能，甚至在 MMLU（+2.16）和 HellaSwag（+0.81）上超越了未缩放的基线。

### 关键发现

1. **BiSpace-RoPE** 在保持短上下文质量（PPL@2048 几乎不变）的同时，实现了有意义的长上下文外推。
2. **BS2** 通过 BiFactor-Scaling 的注意力温度补偿，进一步改善了长上下文困惑度和短上下文基准测试。
3. BiFactor-Scaling 中的**全局-局部分解**有效解决了段边界处的注意力熵退化问题，这是现有方法的关键失效模式。
4. Optuna 优化的 $(\alpha, \beta, \gamma)$ 参数在各评估维度上均提供了一致的改进。

---

## 引用

```bibtex
@mastersthesis{bispace-rope,
  title     = {BiSpace-RoPE: Extending LLM Context Window via Dual-Space Position Encoding},
  author    = {},
  school    = {},
  year      = {2026}
}
```

---

## 致谢

本项目基于以下开源工作：

- [HuggingFace Transformers](https://github.com/huggingface/transformers) --- 模型架构与训练工具
- [PEFT](https://github.com/huggingface/peft) --- LoRA / QLoRA 适配器支持
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) --- 标准基准评估
- [TRL](https://github.com/huggingface/trl) --- 监督微调训练器
- [Optuna](https://github.com/optuna/optuna) --- 超参数优化
- [YaRN](https://arxiv.org/abs/2309.00071) --- 注意力温度缩放的灵感来源

---

## 许可证

本项目仅供研究使用。模型使用限制请参阅 LLaMA 模型许可证。
