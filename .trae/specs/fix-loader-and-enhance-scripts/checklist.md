# Checklist

## 核心修复验证
- [x] `load_model()` 函数已重构为三个完全独立的分支
- [x] 分支 1（base_adapter_path）是自包含的完整流程，不依赖任何公共代码
- [x] 分支 2（adapter_path）是自包含的完整流程
- [x] 分支 3（base model）是自包含的完整流程
- [x] **关键**：不存在重复的模型加载操作（每个分支只调用一次 `from_pretrained()`）
- [x] **关键**：分支 1 正确支持同时使用 `--base-adapter-path` + `--rope-type` + `--adapter-path`
- [x] 代码包含清晰的注释和文档字符串

## 功能正确性验证
### 分支 1: 基础适配器模式
- [x] 从基础适配器加载配置成功
- [x] 当指定 --rope-type 时，能正确覆盖基础适配器的 RoPE 配置
- [x] 模型权重只加载一次
- [x] 基础适配器 LoRA 权重合并成功
- [x] 可选的 LoRA 微调适配器（--adapter-path）能正确二次合并
- [x] 梯度检查点功能正常

### 分支 2: 传统适配器模式
- [x] 从适配器加载完整配置成功
- [x] 忽略 CLI 的 --rope-type 参数（打印警告）
- [x] 模型加载和 LoRA 合并正常
- [x] **向后兼容性**：与修改前的行为完全一致

### 分支 3: 基础模型模式
- [x] 从预训练模型加载默认配置
- [x] 根据 CLI 参数构建 RoPE 配置
- [x] 模型正常加载
- [x] **向后兼容性**：与修改前的行为完全一致

## Shell 脚本增强验证
### continued_pretrain.sh
- [x] 支持 BASE_ADAPTER_PATH 配置变量
- [x] 能正确传递 --base-adapter-path 给 continued_pretrain.py
- [x] 提供清晰的使用示例和注释
- [x] 不影响现有功能（向后兼容）

### finetune.sh
- [x] 支持 BASE_ADAPTER_PATH 配置变量
- [x] 能正确传递参数给 finetune.py
- [x] 文档完善

### eval.sh / eval1.sh / eval2.sh
- [x] 支持混合模式评估（RoPE 方法 + 适配器路径 + 基础适配器组合）
- [x] BASE_COMBOS 数组配置正确
- [x] METHODS 构建逻辑支持三种模式
- [x] 向后兼容（旧配置文件无需修改）

### entropy.sh
- [x] 支持基础适配器熵值评估
- [x] 参数传递正确

### test.sh
- [x] 支持基础适配器性能测试
- [x] 参数传递正确

### search_params.sh
- [x] 现有实现经过验证无 bug
- [x] 配置示例完整且准确

## 集成测试场景验证
- [x] 场景 A: `bash search_params.sh` 使用 base adapter 搜索参数 ✓
- [x] 场景 B: `bash continued_pretrain.sh` 在 base adapter 上继续训练 ✓
- [x] 场景 C: `bash finetune.sh` 在 base adapter 上微调 ✓
- [x] 场景 D: `bash eval.sh` 评估 base adapter 组合模型 ✓
- [x] 场景 E: `bash entropy.sh` 计算 base adapter 熵值 ✓
- [x] 场景 F: 传统模式（所有脚本，不使用新参数）仍然正常 ✓
