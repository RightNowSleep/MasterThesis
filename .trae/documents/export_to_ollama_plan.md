# Plan: 创建 export_to_ollama.py 实现完整自定义 RoPE 模型导出到 Ollama

## 1. 目标与范围

创建一个完整的导出流水线，将本项目支持的 20+ 种自定义 RoPE 模型导出为 Ollama 可用的格式。核心挑战在于：llama.cpp / GGUF 原生只支持 linear, ntk, yarn 等少数 RoPE 类型，而本项目的大量自定义 RoPE（layer-aware, block-layered, inverse-dual 等）在标准 GGUF 中无对应元数据。

## 2. 核心策略：方案 C（权重烘焙 + 元数据保留）+ 方案 B（Python 自定义后端）混合

经过对代码库的深入调研，决定采用以下混合策略，以在**不修改 llama.cpp C++ 源码**的前提下，实现对全部自定义 RoPE 的完整支持：

### 2.1 阶段 1：模型加载与权重烘焙（静态模式）
- 复用 `models/model_loader.py` 的 `load_model()` 和 `load_tokenizer()`，支持三种加载模式（Base Adapter / Traditional Adapter / Base Model）。
- 对于 **static 模式**（`dynamic=False`，即 `rope_scaling` 含 `factor`）：
  - 将模型加载到内存后，遍历所有 attention 层的 `rotary_emb`。
  - 调用每个 `rotary_emb.forward()` 预计算 `cos_cached` 和 `sin_cached` 到 `max_position_embeddings`。
  - 将这些缓存张量作为**额外的只读权重**导出到 Safetensors / PyTorch 状态字典中。
  - 在导出后的模型中，**完全绕过 llama.cpp 的 RoPE 计算**，改为在自定义后端中直接加载并使用预计算的 cos/sin 缓存。
- 对于 **dynamic 模式**（`dynamic=True`）：
  - 由于 scaling factor 在运行时变化，无法预计算单一缓存。
  - 此时需要将**足够的重建参数**保存到 GGUF 自定义元数据中，并在 Python 自定义后端中实时重建 RoPE 计算。

### 2.2 阶段 2：自定义 GGUF 元数据扩展
- 创建 `models/gguf_custom.py`，封装 GGUF 元数据读写逻辑。
- 扩展的元数据字段：
  - `custom_rope.type` (string): 原始 RoPE 类型，如 `"inverse-dual-rope-scaled"`
  - `custom_rope.dynamic` (bool): 是否动态模式
  - `custom_rope.factor` (float): 静态 scaling factor
  - `custom_rope.original_max_position_embeddings` (int): 原始上下文长度
  - `custom_rope.alpha`, `custom_rope.beta`, `custom_rope.gamma` (float): inverse-dual 系列参数
  - `custom_rope.layer_idx` (int, per-layer): 层索引（用于 layer-aware 方法）
  - `custom_rope.i_star` (int): critical dimension index（用于 dual/inverse-dual/block-layered 等）
  - `custom_rope.block_sizes` (array<float>): per-dimension block sizes（用于 block-layered/freq-smooth/freq-reciprocal）
  - `custom_rope.scales` (array<dict>): multi-scale 子空间配置（用于 my-rope2）
  - `custom_rope.attn_scale_coef` (float): attention scaling coefficient
- 对于静态模式，额外保存烘焙后的缓存张量：
  - `rope_cos_cache.layer_{i}` (float32 tensor, shape [max_position_embeddings, head_dim])
  - `rope_sin_cache.layer_{i}` (float32 tensor, shape [max_position_embeddings, head_dim])

### 2.3 阶段 3：llama.cpp 适配（最小侵入）
- **不修改 llama.cpp C++ 源码**，而是采用以下策略：
  - 使用标准 `convert_hf_to_gguf.py` 将模型权重（包括烘焙的 cos/sin 缓存）转换为 GGUF。
  - 在 GGUF 中通过自定义元数据（`custom_rope.*`）标注模型使用了自定义 RoPE。
  - 标准 llama.cpp 可以加载这些权重并忽略未知元数据；实际的 RoPE 计算由 Python 后端接管。

### 2.4 阶段 4：Python 自定义推理后端（`models/ollama_backend.py`）
- 创建一个兼容 Ollama 的 Python 推理服务：
  - 使用 `llama-cpp-python` 或 `ctransformers` 加载 GGUF 权重，执行底层的 transformer 层计算（MLP, Norm, Attention 的 Q/K/V/O projection）。
  - **在 Attention 的 RoPE 应用阶段**，不调用底层 C++ 的 RoPE，而是从 GGUF 中加载预计算的 cos/sin 缓存（静态模式），或使用保存的参数实时计算（动态模式）。
  - 通过 Ollama 的 **自定义后端机制**（创建自定义的 Ollama model runner）或 **OpenAI 兼容 API** 暴露服务。
- 后端架构：
  - `CustomRoPEBackend`：负责加载 GGUF，识别 `custom_rope` 元数据，初始化对应的 Python RoPE 类（复用 `pe_llama.py` 中的类）。
  - `OllamaModelRunner`：包装推理循环，接收 prompt，执行 tokenization → forward → sampling → detokenization。

### 2.5 阶段 5：Modelfile 生成与 Ollama 导入
- `export_to_ollama.py` 在导出完成后，生成标准 Ollama `Modelfile`：
  - `FROM ./model.gguf`
  - `PARAMETER temperature ...`
  - `TEMPLATE ...`
  - `SYSTEM ...`
  - 添加自定义注释标注 `custom_rope` 类型和参数。
- 执行 `ollama create` 命令导入模型。
- 对于使用 Python 自定义后端的模型，Modelfile 中需要指定自定义 runner 路径。

## 3. 文件结构

```
models/
  model_loader.py              # 已有，复用
  export_to_ollama.py          # 新建，主导出脚本（CLI入口）
  ollama_backend.py            # 新建，自定义 Ollama 推理后端
  gguf_custom.py               # 新建，自定义 GGUF 元数据扩展 + 缓存烘焙
  pe_llama.py                  # 已有，RoPE实现（后端复用）
  modeling_llama.py            # 已有，模型结构
  configuration_llama.py       # 已有，配置
```

## 4. 实现步骤

### Step 1: `models/gguf_custom.py`
- 实现 `CustomGGUFWriter`：继承/包装 `gguf.GGUFWriter`，支持写入自定义 `custom_rope.*` 元数据。
- 实现 `bake_rope_caches(model, max_seq_len) -> dict`：
  - 遍历 `model.model.layers`，对每个 `self_attn.rotary_emb` 调用 `forward(dummy_input, seq_len=max_seq_len)` 获取 cos, sin。
  - 返回 `{f"rope_cos_cache.layer_{i}": cos, f"rope_sin_cache.layer_{i}": sin}`。
- 实现 `extract_rope_metadata(config, model) -> dict`：
  - 从 `config.rope_scaling` 和模型结构中抽取所有需要保存的自定义参数。

### Step 2: `models/ollama_backend.py`
- 实现 `CustomRoPEBackend`：
  - `load(gguf_path)`：使用 `gguf` 库读取元数据和权重。
  - 如果检测到 `custom_rope.type`：
    - 静态模式：加载烘焙的 cos/sin 缓存，在 attention 中直接应用。
    - 动态模式：使用保存的参数（i_star, block_sizes, alpha/beta/gamma 等）实例化对应的 `pe_llama.py` 类，在每次 forward 时实时计算。
  - `forward(tokens)`：执行完整的 transformer forward pass，使用自定义 RoPE。
- 实现 `OllamaCompatibleServer`：
  - 提供 OpenAI 兼容的 `/v1/chat/completions` API。
  - 内部调用 `CustomRoPEBackend` 进行推理。

### Step 3: `models/export_to_ollama.py`
- CLI 参数设计（与 `model_loader.py` 保持一致）：
  - `--model-name`, `--base-adapter-path`, `--adapter-path`
  - `--rope-type`, `--rope-factor`, `--rope-dynamic`, `--rope-alpha`, `--rope-beta`, `--rope-gamma`
  - `--output-dir`, `--ollama-model-name`
  - `--quant` (Q4_0, Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0, F16, F32)，默认 Q4_K_M
  - `--max-length`（决定烘焙缓存的长度）
- 导出流程：
  1. 调用 `load_model(args)` 和 `load_tokenizer(args)` 加载模型和分词器。
  2. 检查 `rope_scaling` 类型：
     - TIER 1（linear, ntk, part-ntk, yarn, none）：可直接使用标准 llama.cpp 导出，无需自定义后端。
     - TIER 2/3（自定义类型）：进入混合导出流程。
  3. 如果静态模式：调用 `bake_rope_caches()` 预计算并附加缓存到模型状态字典。
  4. 保存为 HuggingFace 格式（`config.json` + `model.safetensors`）。
  5. 调用 `convert_hf_to_gguf.py` 转换为 GGUF（支持量化）。
     - 需要确保转换脚本能识别并保留自定义元数据和额外张量。
  6. 使用 `CustomGGUFWriter` 追加 `custom_rope.*` 元数据到 GGUF。
  7. 生成 `Modelfile`。
  8. 执行 `ollama create`。
  9. 对于 TIER 2/3，输出使用说明（需要启动 `ollama_backend.py` 作为自定义 runner）。

### Step 4: 测试与验证
- 对每个 TIER 的代表性类型进行端到端测试：
  - TIER 1: `yarn`（标准路径）
  - TIER 2: `block-layered-scaled`（静态烘焙）
  - TIER 3: `inverse-dual-rope-scaled`（静态烘焙 + 动态回退）
- 验证：
  - 导出后的 GGUF 文件大小合理（缓存增加约 `num_layers * max_seq_len * head_dim * 2 * 4 bytes`）。
  - Ollama 可以加载模型（TIER 1 直接加载；TIER 2/3 通过自定义后端加载）。
  - 推理输出与原始 PyTorch 模型一致（数值对比）。

## 5. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 是否修改 llama.cpp C++ | **否** | 避免维护 fork，降低复杂度；通过 Python 后端接管 RoPE |
| 静态模式 RoPE 处理 | **权重烘焙** | 预计算 cos/sin 缓存作为张量保存，推理时零开销 |
| 动态模式 RoPE 处理 | **参数保存 + Python 实时计算** | 无法预计算，保存足够参数在 Python 后端重建原类 |
| 量化支持 | **全精度缓存 + 量化权重** | 缓存使用 F32 保证精度；权重按用户选择量化 |
| Ollama 集成方式 | **自定义 runner / OpenAI API** | 绕过 Ollama 默认的 llama.cpp runner，使用 Python 后端 |

## 6. 风险与回退

- **风险 1**: `convert_hf_to_gguf.py` 可能丢弃未知张量或元数据。
  - **缓解**: 在 `gguf_custom.py` 中直接操作 GGUF 文件，使用 `gguf` 库在转换后追加元数据和张量。
- **风险 2**: 烘焙的缓存导致 GGUF 文件过大。
  - **缓解**: 对于 32 层、8192 长度、128 head_dim 的模型，缓存约 32 * 8192 * 128 * 2 * 4 = 256 MB，可接受。提供 `--max-length` 参数让用户控制。
- **风险 3**: Python 后端推理性能不足。
  - **缓解**: 仅 RoPE 计算在 Python 中执行，底层矩阵运算仍通过 `llama-cpp-python` 的 C++ 内核；对于静态模式，RoPE 仅为查表操作，开销极小。
- **风险 4**: Ollama 不支持自定义 runner。
  - **缓解**: 提供独立的 `ollama_backend.py` 服务，通过 OpenAI 兼容 API 暴露，用户可直接使用而不依赖 Ollama 的 runner 机制。

## 7. 完成标准

- [ ] `models/export_to_ollama.py` 可成功导出所有 TIER 1 类型到 Ollama（标准路径）。
- [ ] `models/export_to_ollama.py` 可成功导出所有 TIER 2/3 静态类型，生成包含烘焙缓存的 GGUF。
- [ ] `models/ollama_backend.py` 可加载 TIER 2/3 的 GGUF 并执行正确推理（与 PyTorch 输出对比误差 < 1e-4）。
- [ ] 生成可用的 `Modelfile` 模板。
- [ ] 提供详细的使用文档和故障排除指南。
