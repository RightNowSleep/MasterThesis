# Checklist

## 模型加载器修改验证
- [x] `models/model_loader.py` 中成功添加了 `--base-adapter-path` 参数定义
- [x] `load_model()` 函数正确实现了基础适配器加载逻辑（配置 + 权重）
- [x] 当同时指定 `base_adapter_path` 和 `rope_type` 时，能正确覆盖 RoPE 配置
- [x] 当同时指定 `base_adapter_path` 和 `adapter_path` 时，两者都能正确加载和合并
- [x] 仅使用现有参数（不使用 base_adapter_path）时，行为与修改前完全一致
- [x] 代码包含清晰的文档字符串和注释

## 参数搜索脚本验证
- [x] `search_attn_scale_params.py` 能正确接收和使用 `--base-adapter-path` 参数
- [x] 使用基础适配器进行 Optuna 搜索时，模型能正确加载
- [x] `set_inverse_dual_rope_scaled_params()` 能正确修改缩放参数
- [x] 搜索过程正常完成，结果保存正确
- [x] 控制台输出清晰显示是否使用了基础适配器及其路径

## Shell 脚本验证
- [x] `search_params.sh` 包含 `BASE_ADAPTER_PATH` 配置变量
- [x] 命令行正确传递 `--base-adapter-path` 参数给 Python 脚本
- [x] 脚本包含详细的使用说明和示例配置
- [x] 脚本头部注释准确描述了新功能的使用方式

## 训练脚本兼容性验证
- [x] `continued_pretrain.py` 无需额外代码修改即可支持新参数
- [x] `finetune.py` 无需额外代码修改即可支持新参数
- [x] 提供了清晰的使用文档或示例说明如何在新场景下使用训练脚本

## 评估脚本兼容性验证
- [x] `eval/perplexity.py` 加载完整适配器模型时工作正常
- [x] `eval/performance.py` 加载完整适配器模型时工作正常
- [x] `eval/eval_harness.py` 加载完整适配器模型时工作正常
- [x] `eval/entropy.py` 加载完整适配器模型时工作正常
- [x] `eval.sh` 中现有的适配器路径配置仍然有效且功能正常
- [x] `entropy.sh` 中现有的适配器路径配置仍然有效且功能正常
- [x] 所有评估脚本的向后兼容性测试通过（不使用新参数时）

## 集成测试场景验证
- [x] 场景1: Base Model → inverse-dual-rope adapter → 搜索 inverse-dual-rope-scaled 参数 ✓
- [x] 场景2: Base Model → dual-rope adapter → 搜索 dual-rope-scaled 参数 ✓
- [x] 场景3: Base Model → yarn adapter → 继续预训练 freq-reciprocal-scaled ✓
- [x] 场景4: 传统方式（无 base_adapter_path）仍然完全可用 ✓
