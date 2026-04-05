# 基于基础RoPE方法的扩展训练和参数搜索 Spec

## Why

当前的 `search_attn_scale_params.py` 和训练脚本直接在 HuggingFace 基础模型（huggyllama/llama-7b）上应用 RoPE 缩放变体（如 inverse-dual-rope-scaled），但这存在问题：

1. **inverse-dual-rope** 将位置编码从一维空间改变为二维空间，**inverse-dual-rope-scaled** 应该在 **inverse-dual-rope** 的基础上进行缩放参数搜索，而不是从零开始
2. 其他 RoPE 变体（yarn、freq-reciprocal-scaled、freq-reciprocal-scaled-no-layer、freq-reciprocal-scaled-adaptive）也存在类似需求，它们都应该在对应的基础 RoPE 方法上进行继续预训练或微调
3. 需要确保所有评估脚本（perplexity、performance、eval_harness、entropy）能够正常工作

## What Changes

### 核心功能变更

1. **新增 `--base-adapter-path` 参数支持**
   - 在模型加载器中支持加载基础适配器（包含基础 RoPE 方法）
   - 在此基础上应用目标 RoPE 缩放类型及其参数

2. **修改 `search_params.sh` 和 `search_attn_scale_params.py`**
   - 使用 `--base-adapter-path` 加载已训练的 inverse-dual-rope 适配器
   - 在该适配器基础上搜索 inverse-dual-rope-scaled 的 alpha/beta/gamma 参数

3. **修改 `continued_pretrain.py` 和 `finetune.py`**
   - 支持 `--base-adapter-path` 参数
   - 允许在基础 RoPE 适配器上进行继续训练或微调
   - 训练时保留基础适配器的 RoPE 配置，仅更新 LoRA 权重

4. **确保评估脚本兼容性**
   - 所有现有评估脚本无需修改即可正常工作
   - 支持通过 `--adapter-path` 加载完整模型（基础 RoPE + 缩放参数 + LoRA 权重）

### 数据流变更

**Before:**
```
Base Model (llama-7b) → Apply RoPE Type (e.g., inverse-dual-rope-scaled) → Train/Evaluate
```

**After:**
```
Base Model (llama-7b)
    ↓
Load Base Adapter (e.g., inverse-dual-rope adapter)
    ↓
Apply Target RoPE Scaling (e.g., inverse-dual-rope-scaled with alpha/beta/gamma)
    ↓
Train/Evaluate
```

## Impact

### Affected specs
- 模型加载流程（model_loader.py）
- 参数搜索脚本（search_attn_scale_params.py）
- 继续预训练脚本（continued_pretrain.py）
- 微调脚本（finetune.py）
- Shell 脚本配置（search_params.sh）

### Affected code
- `/home/linzhen/workspace/MasterThesis/models/model_loader.py` - 核心修改点
- `/home/linzhen/workspace/MasterThesis/search_attn_scale_params.py` - 参数搜索逻辑
- `/home/linzhen/workspace/MasterThesis/search_params.sh` - Shell 脚本配置
- `/home/linzhen/workspace/MasterThesis/continued_pretrain.py` - 继续预训练支持
- `/home/linzhen/workspace/MasterThesis/finetune.py` - 微调支持
- `/home/linzhen/workspace/MasterThesis/models/configuration_llama.py` - 可能需要验证

### 不受影响的代码
- `/home/linzhen/workspace/MasterThesis/eval/perplexity.py` - 已支持 --adapter-path
- `/home/linzhen/workspace/MasterThesis/eval/performance.py` - 已支持 --adapter-path
- `/home/linzhen/workspace/MasterThesis/eval/eval_harness.py` - 已支持 --adapter-path
- `/home/linzhen/workspace/MasterThesis/eval/entropy.py` - 应该已支持
- `/home/linzhen/workspace/MasterThesis/eval.sh` - 已使用适配器路径
- `/home/linzhen/workspace/MasterThesis/entropy.sh` - 已使用适配器路径

## ADDED Requirements

### Requirement: 支持基础适配器加载

系统应提供 `--base-adapter-path` 参数，允许用户指定一个包含基础 RoPE 方法的适配器路径。当提供此参数时：
1. 从基础适配器加载模型权重和 RoPE 配置
2. 在此基础上应用目标 RoPE 缩放类型（如果指定了 `--rope-type`）
3. 如果同时指定了 `--adapter-path`，则在最后合并 LoRA 适配器

#### Scenario: 参数搜索时使用基础适配器
- **WHEN** 用户运行 `search_attn_scale_params.py` 并指定 `--base-adapter-path finetunes/continued_pretrain/inverse-dual-rope_xxx --rope-type inverse-dual-rope-scaled`
- **THEN** 系统应该：
  - 加载 inverse-dual-rope 适配器的权重和配置
  - 应用 inverse-dual-rope-scaled 的 alpha/beta/gamma 参数
  - 在此基础上进行 Optuna 参数搜索

#### Scenario: 继续预训练时使用基础适配器
- **WHEN** 用户运行 `continued_pretrain.py` 并指定 `--base-adapter-path finetunes/continued_pretrain/inverse-dual-rope_xxx --rope-type inverse-dual-rope-scaled`
- **THEN** 系统应该：
  - 加载 inverse-dual-rope 适配器作为起始点
  - 使用 inverse-dual-rope-scaled 作为当前训练的 RoPE 类型
  - 仅训练新的 LoRA 权重，保留基础适配器的权重不变

### Requirement: 向后兼容性

所有现有调用方式必须保持不变：
- 不指定 `--base-adapter-path` 时，行为与之前完全一致
- 现有的 `--adapter-path` 功能不受影响
- 所有评估脚本无需任何修改即可正常工作

#### Scenario: 传统方式仍然有效
- **WHEN** 用户按原有方式运行（不使用 `--base-adapter-path`）
- **THEN** 系统行为与修改前完全一致

## MODIFIED Requirements

### Requirement: 模型加载流程增强

修改 `models/model_loader.py` 中的 `load_model()` 函数，增加以下逻辑：

```python
if args.base_adapter_path:
    # 1. 从基础适配器加载配置
    config = LlamaConfig.from_pretrained(args.base_adapter_path)
    # 2. 如果指定了新的 rope_type，覆盖配置中的 RoPE 设置
    if args.rope_type != "none":
        rope_scaling = _build_rope_scaling(args)
        config.rope_scaling = rope_scaling
    # 3. 加载基础模型
    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        ...
    )
    # 4. 合并基础适配器权重
    model = PeftModel.from_pretrained(model, args.base_adapter_path)
    model = model.merge_and_unload()
elif args.adapter_path:
    # 现有逻辑保持不变
    ...
else:
    # 现有逻辑保持不变
    ...
```

### Requirement: 参数搜索脚本修改

修改 `search_attn_scale_params.py`：
1. 新增 `--base-adapter-path` CLI 参数
2. 将其传递给 `load_model()`
3. 更新 `set_inverse_dual_rope_scaled_params()` 函数注释说明

### Requirement: Shell 脚本配置更新

修改 `search_params.sh`：
1. 添加 `BASE_ADAPTER_PATH` 配置变量
2. 在命令行中传入 `--base-adapter-path` 参数
3. 提供示例配置注释

### Requirement: 训练脚本增强

修改 `continued_pretrain.py` 和 `finetune.py`：
1. 通过 `add_args_model()` 自动获得 `--base-adapter-path` 参数（已在 model_loader 中定义）
2. 无需额外修改，因为 load_model() 会处理该参数

## REMOVED Requirements

无移除的需求。这是一个纯增量功能，不影响现有能力。
