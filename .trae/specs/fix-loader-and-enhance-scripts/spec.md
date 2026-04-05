# 修复模型加载器逻辑错误并增强 Shell 脚本支持 Spec

## Why

当前 `models/model_loader.py` 的 `load_model()` 函数存在严重的逻辑缺陷：
1. **分支 1（base_adapter_path 模式）会在公共代码中重复加载模型**，导致已合并的基础适配器权重丢失
2. **代码结构混乱**：三个分支的处理不一致，部分操作重复执行
3. **Shell 脚本未完全支持新功能**：大部分 Shell 脚本只支持传统的两种模式，缺少对 `--base-adapter-path` 的支持

## What Changes

### 核心修复：重构 load_model() 函数
- **BREAKING**: 重构三分支逻辑为独立的完整流程，每个分支内部完成所有必要操作
- 分支 1（base_adapter）：独立完成配置→加载模型→合并基础适配器→可选合并 LoRA → 返回
- 分支 2（adapter）：独立完成配置→加载模型→合并 LoRA → 返回
- 分支 3（base）：独立完成配置→加载模型 → 返回
- 消除公共代码导致的重复操作

### 增强 Shell 脚本支持
为以下脚本添加 `--base-adapter-path` 参数传递能力：
- `continued_pretrain.sh`: 支持基于基础适配器的继续预训练
- `finetune.sh`: 支持基于基础适配器的微调
- `entropy.sh`: 支持评估基础适配器+目标缩放的组合
- `eval.sh`, `eval1.sh`, `eval2.sh`: 支持评估组合模型
- `test.sh`: 支持测试组合模型
- `search_params.sh`: 已支持（验证并完善）

## Impact
- Affected specs: add-base-adapter-support（修正其实现错误）
- Affected code:
  - `/home/linzhen/workspace/MasterThesis/models/model_loader.py` - 核心修复
  - 所有 Shell 脚本 - 增强参数支持

## ADDED Requirements

### Requirement: 正确的三分支模型加载逻辑

系统 SHALL 提供三种互斥且完整的模型加载模式：

#### Scenario A: 基础适配器模式（Base Adapter Mode）
- **WHEN** 用户指定 `--base-adapter-path` （可同时指定 `--rope-type` 和可选的 `--adapter-path`）
- **THEN** 系统应该：
  1. 从基础适配器路径加载配置
  2. 如果指定了 `--rope-type`，覆盖 RoPE 配置
  3. 设置运行时参数（max_length, use_cache, dtype, quantization）
  4. 加载基础模型权重
  5. 合并基础适配器 LoRA 权重
  6. 如果指定了 `--adapter-path`，再合并 LoRA 微调权重
  7. 应用梯度检查点（如果启用）
  8. 返回最终模型（**不经过任何公共重复代码**）

#### Scenario B: 传统适配器模式（Adapter Mode）
- **WHEN** 用户仅指定 `--adapter-path`（不指定 base_adapter_path）
- **THEN** 系统应该：
  1. 从适配器路径加载完整配置（忽略 --rope-type）
  2. 设置运行时参数
  3. 加载基础模型权重
  4. 合并 LoRA 适配器权重
  5. 应用梯度检查点
  6. 返回最终模型

#### Scenario C: 基础模型模式（Base Model Mode）
- **WHEN** 用户不指定任何适配器路径
- **THEN** 系统应该：
  1. 从预训练模型加载默认配置
  2. 根据 CLI 参数构建 RoPE 配置
  3. 设置运行时参数
  4. 加载模型权重
  5. 应用梯度检查点
  6. 返回最终模型

### Requirement: Shell 脚本的完整参数传递

每个 Shell 脚本 SHALL 能够透明地传递 `--base-adapter-path` 参数给底层 Python 脚本：

- **continued_pretrain.sh**: 支持 `BASE_ADAPTER_PATH` 变量，用于在已有 RoPE 方法上继续训练新变体
- **finetune.sh**: 支持 `BASE_ADAPTER_PATH` 变量，用于在已有方法上微调新变体
- **eval.sh / eval1.sh / eval2.sh**: 支持混合模式评估（RoPE 方法列表 + 适配器列表 + 基础适配器组合）
- **entropy.sh**: 支持基础适配器熵值评估
- **test.sh**: 支持基础适配器性能测试
- **search_params.sh**: 验证现有实现的正确性

## MODIFIED Requirements

### Requirement: load_model() 函数重构

将现有的有缺陷的实现替换为清晰的**三路独立分支**结构：

```python
def load_model(args, quantization_config=None):
    print(f"Loading model : {args.model_name}")
    
    # ═══ 分支 1: 基础适配器模式 ═══
    if getattr(args, "base_adapter_path", None):
        # 1. 加载配置（从基础适配器）
        # 2. 可选：覆盖 RoPE 配置
        # 3. 设置运行时参数
        # 4. 处理量化
        # 5. 加载模型
        # 6. 合并基础适配器
        # 7. 可选：合并 LoRA 适配器
        # 8. 可选：梯度检查点
        # 9. 返回（不进入其他分支）
        
    # ═══ 分支 2: 传统适配器模式 ═══
    elif args.adapter_path:
        # 1. 加载配置（从适配器）
        # 2. 设置运行时参数
        # 3. 处理量化
        # 4. 加载模型
        # 5. 合并 LoRA 适配器
        # 6. 可选：梯度检查点
        # 7. 返回
        
    # ═══ 分支 3: 基础模型模式 ═══
    else:
        # 1. 加载配置（从预训练模型）
        # 2. 构建 RoPE 配置
        # 3. 设置运行时参数
        # 4. 处理量化
        # 5. 加载模型
        # 6. 可选：梯度检查点
        # 7. 返回
```

**关键原则**：
- 每个分支是**自包含的完整流程**
- 不存在"公共代码"导致重复执行
- 代码清晰易读易维护

## REMOVED Requirements

### Requirement: 有缺陷的公共代码模式
**原因**: 当前的实现在 if/elif/else 之后有公共代码块，导致：
- 分支 1 重复加载模型（严重 bug）
- 配置参数被多次设置
- 难以维护和理解

**迁移**: 替换为每个分支独立完成的清晰结构
