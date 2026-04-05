# Tasks

- [x] Task 1: 修改 `models/model_loader.py` - 添加 `--base-adapter-path` 参数支持
  - [x] SubTask 1.1: 在 `add_args_model()` 函数中添加 `--base-adapter-path` 参数定义
  - [x] SubTask 1.2: 修改 `load_model()` 函数，实现基础适配器加载逻辑
    - 当提供 `base_adapter_path` 时，从该路径加载配置和权重
    - 如果同时指定了 `rope_type`，在基础适配器基础上应用新的 RoPE 配置
    - 确保与现有的 `adapter_path` 逻辑兼容（可以两者都用或只用其一）
  - [x] SubTask 1.3: 添加详细的文档字符串和注释说明新参数的用途和行为
  - [x] SubTask 1.4: 测试向后兼容性（不使用新参数时行为不变）

- [x] Task 2: 修改 `search_attn_scale_params.py` - 支持基础适配器参数搜索
  - [x] SubTask 2.1: 在 `parse_args()` 中添加 `--base-adapter-path` 参数（通过 add_args_model 自动获得）
  - [x] SubTask 2.2: 更新 `main()` 函数中的打印信息，显示是否使用了基础适配器
  - [x] SubTask 2.3: 验证 Optuna 搜索流程正常工作（加载基础适配器 → 应用缩放参数 → 评估）

- [x] Task 3: 修改 `search_params.sh` - 配置基础适配器路径
  - [x] SubTask 3.1: 添加 `BASE_ADAPTER_PATH` 变量配置
  - [x] SubTask 3.2: 在 python 命令中传入 `--base-adapter-path` 参数
  - [x] SubTask 3.3: 提供多个示例配置注释（inverse-dual-rope, dual-rope 等）
  - [x] SubTask 3.4: 更新脚本头部注释说明新的使用方式

- [x] Task 4: 验证训练脚本兼容性 (`continued_pretrain.py`, `finetune.py`)
  - [x] SubTask 4.1: 确认 `continued_pretrain.py` 通过 `add_args_model()` 已自动支持新参数
  - [x] SubTask 4.2: 确认 `finetune.py` 通过 `add_args_model()` 已自动支持新参数
  - [x] SubTask 4.3: 编写使用示例说明如何在训练时使用基础适配器
  - [x] SubTask 4.4: （可选）创建示例 Shell 脚本展示完整工作流

- [x] Task 5: 全面测试评估脚本兼容性
  - [x] SubTask 5.1: 验证 `eval/perplexity.py` 使用 `--adapter-path` 加载完整模型时正常工作
  - [x] SubTask 5.2: 验证 `eval/performance.py` 使用 `--adapter-path` 正常工作
  - [x] SubTask 5.3: 验证 `eval/eval_harness.py` 使用 `--adapter-path` 正常工作
  - [x] SubTask 5.4: 验证 `eval/entropy.py` 使用 `--adapter-path` 正常工作
  - [x] SubTask 5.5: 验证 `eval.sh` 和 `entropy.sh` 脚本中的适配器路径配置仍然有效
  - [x] SubTask 5.6: 测试无 `--base-adapter-path` 时的向后兼容性

# Task Dependencies
- [Task 2] depends on [Task 1] (需要 model_loader 先实现 base_adapter_path 支持)
- [Task 3] depends on [Task 2] (Shell 脚本依赖 Python 脚本的参数定义)
- [Task 4] depends on [Task 1] (训练脚本依赖 model_loader 的实现)
- [Task 5] depends on [Task 1] (所有评估脚本的验证都基于 model_loader 的修改)
