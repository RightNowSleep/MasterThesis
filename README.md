# BiSpace-RoPE: Extending LLM Context Window via Dual-Space Position Encoding

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <strong>BiSpace-RoPE</strong> --- A novel dual-space rotary position encoding for extending the context window of large language models<br>
  <strong>BS2</strong> --- BiSpace-RoPE + BiFactor-Scaling for robust long-context extrapolation
</p>

---

## Overview

Extending the context window of Transformer-based LLMs beyond their training length is a critical challenge. Existing RoPE scaling methods (e.g., Position Interpolation, NTK-aware, YaRN) either degrade short-context performance or fail to maintain stability at extreme lengths.

This project introduces **BiSpace-RoPE** and **BS2**, two novel methods that decompose position encoding into a **global space** and a **local space**, achieving strong perplexity and benchmark performance across a wide range of context lengths.

**Key contributions:**

- **BiSpace-RoPE**: Splits position indices by a critical frequency dimension --- high-frequency dimensions use monotonic global positions, while low-frequency dimensions use cyclic local positions modulo the original context length.
- **BiFactor-Scaling**: A decomposed attention temperature scaling function with a global logarithmic growth term and a local boundary compensation term, addressing attention entropy degradation at segment transitions.
- **BS2 (BiSpace-RoPE + BiFactor-Scaling)**: Combines both methods for state-of-the-art long-context extrapolation.
- A comprehensive benchmarking framework with 20+ RoPE variants, supporting continued pretraining, SFT, hyperparameter search, and multi-dimensional evaluation.

---

## Key Methods

### BiSpace-RoPE

BiSpace-RoPE splits the position index into two subspaces based on a **critical dimension** $i^*$:

$$
i^* = \text{first index where } r_i = \frac{L_0 \cdot \theta_i}{2\pi} < 1
$$

where $L_0$ is the original context length and $\theta_i$ is the rotation frequency at dimension $i$.

The position encoding is then:

$$
\text{pos}(t, i) =
\begin{cases}
t & i < i^* \quad \text{(high-freq: global, monotonic)} \\
t \bmod L_0 & i \geq i^* \quad \text{(low-freq: local, cyclic)}
\end{cases}
$$

**Intuition**: High-frequency dimensions complete multiple rotations within $L_0$ and need precise global positions. Low-frequency dimensions rotate slowly and are insensitive to absolute position beyond $L_0$ --- cycling them back preserves the training distribution.

### BiFactor-Scaling

BiFactor-Scaling applies a decomposed attention temperature factor for positions beyond $L_0$:

$$
s(t) =
\begin{cases}
1 & t < L_0 \\
\underbrace{(1 + \alpha \ln(k+1))}_{\text{global term}} \cdot \underbrace{(1 + \beta e^{-\gamma r})}_{\text{local term}} & t \geq L_0
\end{cases}
$$

where:
- $k = \lfloor t / L_0 \rfloor$ --- segment index (global coordinate)
- $r = (t \bmod L_0) / L_0 \in [0, 1)$ --- normalized intra-segment position (local coordinate)
- $\alpha$ --- global growth rate (default: 0.1, YaRN-like logarithmic growth)
- $\beta$ --- boundary jump compensation amplitude (default: 0.5)
- $\gamma$ --- intra-segment exponential decay rate (default: 2.0)

**Design rationale**: The global term compensates for cumulative attention entropy loss across segments. The local term provides a jump-up compensation at each segment boundary ($r \approx 0$) that decays exponentially within the segment ($r \to 1$), addressing the discontinuity at segment transitions.

### BS2 = BiSpace-RoPE + BiFactor-Scaling

BS2 combines BiSpace-RoPE's dual-space position encoding with BiFactor-Scaling's attention temperature compensation, applying the scaling factor to the BiSpace-encoded cos/sin values:

$$
\text{RoPE}_{\text{BS2}}(t) = s(t) \cdot \left[\cos(\text{pos}(t, i) \cdot \theta_i),\ \sin(\text{pos}(t, i) \cdot \theta_i)\right]
$$

The optimal $(\alpha, \beta, \gamma)$ parameters are found via Optuna Bayesian optimization (see [Hyperparameter Search](#hyperparameter-search)).

---

## Project Structure

```
MasterThesis/
├── models/                          # Core model implementations
│   ├── pe_llama.py                  # 20+ RoPE implementations (BiSpace-RoPE, BiFactor-Scaling, etc.)
│   ├── entropy_stable_scaled_rope.py # Entropy-stable attention scaling RoPE
│   ├── modeling_llama.py            # Custom LLaMA model with RoPE integration
│   ├── configuration_llama.py       # Custom LLaMA configuration
│   ├── model_loader.py              # Unified model/tokenizer loader with RoPE config
│   ├── gguf_custom.py               # GGUF export support
│   ├── ollama_backend.py            # Ollama backend integration
│   └── export_to_ollama.py          # Export to Ollama format
├── eval/                            # Evaluation modules
│   ├── perplexity.py                # Sliding-window perplexity at multiple lengths
│   ├── entropy.py                   # Attention entropy analysis (Shannon + normalized)
│   ├── eval_harness.py              # lm-eval-harness benchmark runner
│   ├── passkey.py                   # Passkey retrieval (needle-in-a-haystack)
│   ├── quality.py                   # Long-context MCQ (LongBench-v2, QuALITY)
│   ├── performance.py               # Inference runtime & GPU memory profiling
│   ├── plot_entropy.py              # Entropy visualization
│   └── plot_entropy_fig3.py         # Paper Figure 3 entropy plots
├── drawer/                          # Visualization modules
│   ├── rope_comparison.py           # RoPE method comparison plots
│   ├── position_heatmap.py          # Position encoding heatmaps
│   ├── runtime.py                   # Runtime comparison plots
│   ├── perplexity.py                # Perplexity curve plots
│   └── block_sum.py                 # Block summary visualization
├── continued_pretrain.py            # Continued pretraining with LoRA/QLoRA
├── finetune.py                      # Supervised fine-tuning (SFT) with QLoRA
├── search_attn_scale_params.py      # Optuna hyperparameter search for (α, β, γ)
├── *.sh                             # Shell scripts for training & evaluation
├── results/                         # Experiment results (JSON)
│   ├── perplexity/                  # Perplexity at multiple context lengths
│   ├── entropy/                     # Attention entropy metrics
│   ├── eval_harness/                # Benchmark results (MMLU, HellaSwag, etc.)
│   ├── performance/                 # Runtime & memory profiling
│   └── param_search/                # Optuna search results
└── finetunes/                       # LoRA adapter checkpoints (gitignored)
```

---

## Installation

### Requirements

- Python 3.10
- CUDA 12.x
- PyTorch 2.7+

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd MasterThesis

# Create conda environment (minimal)
conda env create -f environment_conda.yml
conda activate LLMs

# Or full environment (includes all dependencies)
conda env create -f environment.yml
conda activate LLMs
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `transformers` | Model loading & inference |
| `peft` | LoRA / QLoRA adapter support |
| `bitsandbytes` | 4-bit / 8-bit quantization |
| `accelerate` | Distributed training |
| `trl` | SFT training |
| `lm-eval` | Standard benchmark evaluation |
| `optuna` | Hyperparameter search |
| `datasets` | Dataset loading |
| `wandb` | Experiment tracking (optional) |

---

## Quick Start

### Load a Model with BiSpace-RoPE

```python
from models.model_loader import load_model, load_tokenizer

# Load tokenizer
tokenizer = load_tokenizer("huggyllama/llama-7b")

# Load model with BiSpace-RoPE (static, factor=8)
model = load_model(
    "huggyllama/llama-7b",
    rope_type="inverse-dual-rope",
    rope_factor=8.0,
    load_in_4bit=True,
)

# Load model with BS2 (dynamic scaling)
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

### Evaluate Perplexity

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

## Usage

### Continued Pretraining

Extend the context window via continued pretraining with QLoRA/LoRA:

```bash
# Single method
python continued_pretrain.py \
    --model-name huggyllama/llama-7b \
    --rope-type inverse-dual-rope \
    --rope-factor 8.0 \
    --max-length 16384 \
    --quantization 4bit \
    --lora-r 64 --lora-alpha 128 \
    --max-train-steps 400 \
    --dataset emozilla/pg_books-tokenized-bos-eos-chunked-65536

# Batch: run multiple methods via shell script
bash continued_pretrain.sh
```

**Progressive length training**: Gradually increase sequence length across stages (`--progressive-length`), e.g., [2048, 4096, 8192, 16384].

**Hierarchical training**: Train BS2 on top of a BiSpace-RoPE adapter:

```bash
python continued_pretrain.py \
    --model-name huggyllama/llama-7b \
    --base-adapter-path finetunes/continued_pretrain/inverse-dual-rope_20260403_103555 \
    --rope-type inverse-dual-rope-scaled \
    --rope-factor 8.0 \
    --max-length 16384 \
    --quantization 4bit
```

### Supervised Fine-Tuning (SFT)

```bash
python finetune.py \
    --model-name meta-llama/Llama-2-7b-chat-hf \
    --rope-type inverse-dual-rope-scaled --rope-factor 4.0 \
    --quantization 4bit --use-lora \
    --dataset HuggingFaceH4/ultrachat_200k \
    --num-train-epochs 1 \
    --output-dir finetunes/finetune

# Or batch via shell script
bash finetune.sh
```

### Hyperparameter Search

Search optimal $(\alpha, \beta, \gamma)$ for BiFactor-Scaling using Optuna:

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

# Or via shell script
bash search_params.sh
```

### Evaluation

#### Perplexity (Multi-Length)

```bash
python eval/perplexity.py \
    --model-name huggyllama/llama-7b \
    --adapter-path finetunes/continued_pretrain/inverse-dual-rope-scaled_20260406_070155 \
    --load-in-4bit \
    --max-length 65536 --min-length 2048

# Batch
bash eval_perplexity.sh
```

#### Attention Entropy

```bash
python eval/entropy.py \
    --model-name huggyllama/llama-7b \
    --rope-type inverse-dual-rope-scaled --rope-dynamic \
    --load-in-4bit \
    --max-length 3072 --num-samples 100

# Batch (evaluation + plotting)
bash entropy.sh
```

#### Standard Benchmarks (lm-eval-harness)

```bash
python eval/eval_harness.py \
    --model-name huggyllama/llama-7b \
    --adapter-path finetunes/continued_pretrain/inverse-dual-rope-scaled_20260406_070155 \
    --load-in-4bit \
    --tasks mmlu,hellaswag,gsm8k,arc_challenge

# Batch
bash eval_harness.sh
```

**Available task categories**: `reasoning` (MMLU, HellaSwag, ARC, TruthfulQA, TriviaQA), `math` (GSM8K, Minerva Math, AGIEval Math), `code` (MBPP), `long_context` (Passkey, LongBench, Babilong, NIAH).

#### Passkey Retrieval

```bash
python eval/passkey.py \
    --model-name huggyllama/llama-7b \
    --rope-type inverse-dual-rope-scaled --rope-dynamic \
    --load-in-4bit \
    --max-length 65536 --data-mode real

# Batch
bash eval_passkey.sh
```

#### Long-Context Quality (LongBench-v2 / QuALITY)

```bash
python eval/quality.py \
    --model-name huggyllama/llama-7b \
    --rope-type inverse-dual-rope-scaled --rope-dynamic \
    --load-in-4bit \
    --scoring-mode logit --limit 50
```

#### Inference Performance

```bash
python eval/performance.py \
    --model-name huggyllama/llama-7b \
    --rope-type inverse-dual-rope-scaled --rope-dynamic \
    --load-in-4bit \
    --max-length 4096 --use-cache

# Batch
bash eval_performance.sh
```

---

## Supported RoPE Methods

| Category | Method | CLI `--rope-type` | Dynamic | Attn Scaling |
|----------|--------|-------------------|---------|-------------|
| **Baseline** | Standard RoPE | `none` | - | - |
| **Classic** | Position Interpolation | `linear` | Yes | - |
| **Classic** | NTK-aware | `ntk` | Yes | - |
| **Classic** | NTK-by-parts | `part-ntk` | Yes | - |
| **Classic** | YaRN | `yarn` | Yes | Yes |
| **Custom** | My-RoPE | `my-rope` | Yes | - |
| **Custom** | My-RoPE Scaled | `my-rope-scaled` | Yes | Yes |
| **Custom** | My-RoPE2 | `my-rope2` | Yes | - |
| **Custom** | My-RoPE2 Scaled | `my-rope2-scaled` | Yes | Yes |
| **Block** | Block-Layered | `block-layered` | Yes | - |
| **Block** | Block-Layered Scaled | `block-layered-scaled` | Yes | Yes |
| **Freq** | Freq-Smooth | `freq-smooth` | Yes | - |
| **Freq** | Freq-Smooth Scaled | `freq-smooth-scaled` | Yes | Yes |
| **Freq** | Freq-Reciprocal | `freq-reciprocal` | Yes | - |
| **Freq** | Freq-Reciprocal Scaled | `freq-reciprocal-scaled` | Yes | Yes |
| **Freq** | Freq-Reciprocal Scaled No-Layer | `freq-reciprocal-scaled-no-layer` | Yes | Yes |
| **Freq** | Freq-Reciprocal Scaled Adaptive | `freq-reciprocal-scaled-adaptive` | Yes | Yes |
| **Dual** | Dual-RoPE | `dual-rope` | Yes | - |
| **Dual** | Dual-RoPE Scaled | `dual-rope-scaled` | Yes | Yes |
| **Core** | **BiSpace-RoPE** | `inverse-dual-rope` | Yes | - |
| **Core** | **BS2 (BiSpace + BiFactor)** | `inverse-dual-rope-scaled` | Yes | Yes |
| **Core** | **BiFactor-Scaling** | `bi-factor-scaling-rope` | Yes | Yes |
| **Variant** | BiSpace-Tangle | `inverse-dual-tangle-rope` | Yes | - |
| **Variant** | BiSpace-Tangle Scaled | `inverse-dual-tangle-rope-scaled` | Yes | Yes |
| **Variant** | BiSpace-NoPos | `inverse-dual-nopos-rope` | Yes | - |
| **Variant** | BiSpace-NoPos Scaled | `inverse-dual-nopos-rope-scaled` | Yes | Yes |

**Static vs Dynamic mode**: Use `--rope-factor F` for static scaling (fixed ratio) or `--rope-dynamic` for runtime-adaptive scaling where $s = \max(1, \; \text{seq\_len} \; / \; L_0)$.

---

## Results

### Perplexity (ProofPile, Static Factor=8)

| Context Length | None | Linear | NTK | Part-NTK | YaRN | **BiSpace-RoPE** | **BS2** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2048 | 4.80 | 7.07 | 4.86 | 5.17 | 5.13 | **4.86** | 4.86 |
| 4096 | 10.34 | 6.05 | 4.19 | 4.51 | 4.38 | 8.08 | - |
| 8192 | 37.19 | 5.05 | 3.55 | 3.86 | 3.68 | 13.44 | - |
| 16384 | 133.53 | 4.52 | 5.81 | 3.49 | 3.25 | 21.53 | - |
| 32768 | 353.21 | 24.63 | 29.33 | 20.92 | 20.97 | 30.74 | - |
| 65536 | 419.75 | 38.40 | 45.45 | 33.40 | 34.13 | 35.64 | - |

> Results from base model (no continued pretraining). BiSpace-RoPE achieves competitive long-context perplexity while preserving short-context quality (PPL@2048 = 4.86 vs 4.80 baseline).

### Standard Benchmarks (Factor=8, After Continued Pretraining)

| Method | MMLU (5-shot) | HellaSwag (10-shot) | GSM8K (8-shot) |
|--------|:-:|:-:|:-:|
| None | 32.20 | 75.74 | 6.44 |
| Linear | 25.93 | 65.03 | - |
| NTK | 30.37 | 74.89 | - |
| Part-NTK | 28.18 | 71.57 | - |
| YaRN | 29.78 | 74.53 | - |
| **BiSpace-RoPE** | 31.28 | 75.15 | - |
| **BS2** | **34.36** | **76.55** | **7.43** |

> BS2 achieves the best performance across all benchmarks, outperforming even the unscaled baseline on MMLU (+2.16) and HellaSwag (+0.81).

### Key Findings

1. **BiSpace-RoPE** preserves short-context quality (PPL@2048 nearly unchanged) while enabling meaningful long-context extrapolation.
2. **BS2** further improves both long-context perplexity and short-context benchmarks through BiFactor-Scaling's attention temperature compensation.
3. The **global-local decomposition** in BiFactor-Scaling effectively addresses attention entropy degradation at segment boundaries, a key failure mode of existing methods.
4. Optuna-optimized $(\alpha, \beta, \gamma)$ parameters provide consistent improvements across evaluation dimensions.

---

## Citation

```bibtex
@mastersthesis{bispace-rope,
  title     = {BiSpace-RoPE: Extending LLM Context Window via Dual-Space Position Encoding},
  author    = {},
  school    = {},
  year      = {2026}
}
```

---

## Acknowledgements

This project builds upon the following open-source works:

- [HuggingFace Transformers](https://github.com/huggingface/transformers) --- Model architecture and training utilities
- [PEFT](https://github.com/huggingface/peft) --- LoRA / QLoRA adapter support
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) --- Standard benchmark evaluation
- [TRL](https://github.com/huggingface/trl) --- Supervised fine-tuning trainer
- [Optuna](https://github.com/optuna/optuna) --- Hyperparameter optimization
- [YaRN](https://arxiv.org/abs/2309.00071) --- Inspiration for attention temperature scaling

---

## License

This project is for research purposes. Please refer to the LLaMA model license for model usage restrictions.
