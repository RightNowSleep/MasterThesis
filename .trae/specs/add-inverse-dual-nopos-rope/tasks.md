# Tasks

- [x] Task 1: 在 `models/pe_llama.py` 中创建 `LlamaInverseDualNoPosRoPEEmbedding` 类
  - [x] 复制 `LlamaInverseDualRoPEEmbedding` 作为模板
  - [x] 修改 `_set_cos_sin_cache`: 将 `pos_2 = t % L_0` 改为 `pos_2 = torch.zeros_like(t)`（低频维度位置恒为 0）
  - [x] 更新 docstring 说明低频维度无位置编码的设计
  - [x] 保持 forward 接口不变

- [x] Task 2: 在 `models/pe_llama.py` 中创建 `LlamaInverseDualNoPosRoPEScaledEmbedding` 类
  - [x] 继承 `LlamaInverseDualNoPosRoPEEmbedding`（而非原来的 scaled 版本）
  - [x] 复制 `LlamaInverseDualRoPEScaledEmbedding` 的 `_compute_attn_scale` 和 `_set_cos_sin_cache` / `forward` 方法
  - [x] 更新 docstring 和类说明

- [x] Task 3: 在 `models/configuration_llama.py` 中注册新类型
  - [x] 在 valid_types 列表中追加 `"inverse-dual-nopos-rope"` 和 `"inverse-dual-nopos-rope-scaled"`
  - [x] 在 `_deprecated_dynamic_map` 中追加对应的 deprecated 映射
  - [x] 在类型特定参数验证区域为 `"inverse-dual-nopos-rope-scaled"` 添加 alpha/beta/gamma 验证（复用现有逻辑）
  - [x] 更新类的 docstring 中支持的类型列表

- [x] Task 4: 在 `models/modeling_llama.py` 的 `_init_rope()` 中添加新分支
  - [x] 添加 `elif scaling_type == "inverse-dual-nopos-rope"` 分支，实例化 `LlamaInverseDualNoPosRoPEEmbedding`
  - [x] 添加 `elif scaling_type == "inverse-dual-nopos-rope-scaled"` 分支，实例化 `LlamaInverseDualNoPosRoPEScaledEmbedding` 并传入 alpha/beta/gamma
  - [x] 更新 ValueError 中的有效类型列表

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 3] 与 [Task 1][Task 2] 可并行
- [Task 4] depends on [Task 1][Task 2][Task 3]
