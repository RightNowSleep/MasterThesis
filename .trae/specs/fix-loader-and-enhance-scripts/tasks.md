# Tasks

- [x] Task 1: **修复 `models/model_loader.py` 的致命 Bug** - 重构 load_model() 函数
  - [x] SubTask 1.1: 分析当前代码的三个分支逻辑，明确每个分支的完整职责
  - [x] SubTask 1.2: 重构分支 1（base_adapter_path 模式）为自包含的完整流程
    - 从基础适配器加载配置
    - 可选覆盖 RoPE 配置
    - 设置运行时参数（max_length, use_cache）
    - 处理量化配置
    - 加载基础模型权重（仅一次！）
    - 合并基础适配器 LoRA 权重
    - 可选：合并 --adapter-path 的 LoRA 权重
    - 可选：启用梯度检查点
    - 打印完成信息并返回
  - [x] SubTask 1.3: 重构分支 2（adapter_path 模式）为自包含的完整流程
    - 从适配器加载完整配置
    - 设置运行时参数
    - 处理量化配置
    - 加载模型权重
    - 合并 LoRA 适配器
    - 可选：梯度检查点
    - 返回
  - [x] SubTask 1.4: 重构分支 3（基础模式）为自包含的完整流程
    - 从预训练模型加载默认配置
    - 构建 RoPE 配置
    - 设置运行时参数
    - 处理量化配置
    - 加载模型权重
    - 可选：梯度检查点
    - 返回
  - [x] SubTask 1.5: 删除所有公共重复代码，确保三个分支完全独立
  - [x] SubTask 1.6: 添加清晰的注释说明每个分支的数据流和适用场景

- [x] Task 2: **增强 `continued_pretrain.sh`** 支持基础适配器参数
  - [x] SubTask 2.1: 在 RoPE Methods Configuration 部分添加 BASE_ADAPTER_PATH 变量
  - [x] SubTask 2.2: 在 run_pretrain() 函数中支持传递 --base-adapter-path 参数
  - [x] SubTask 2.3: 更新脚本头部注释说明新功能
  - [x] SubTask 2.4: 提供使用示例注释

- [x] Task 3: **增强 `finetune.sh`** 支持基础适配器参数
  - [x] SubTask 3.1: 添加 BASE_ADAPTER_PATH 配置变量和示例
  - [x] SubTask 3.2: 修改 run_finetune() 函数传递新参数
  - [x] SubTask 3.3: 更新文档和使用说明

- [x] Task 4: **增强评估类 Shell 脚本** (eval.sh, eval1.sh, eval2.sh)
  - [x] SubTask 4.1: 为 eval.sh 添加 BASE_COMBOS 数组配置
  - [x] SubTask 4.2: 为 eval1.sh 添加相同支持
  - [x] SubTask 4.3: 为 eval2.sh 添加相同支持
  - [x] SubTask 4.4: 修改各脚本的 METHODS 构建逻辑，支持三种模式的混合使用
  - [x] SubTask 4.5: 确保向后兼容（不设置新参数时行为不变）

- [x] Task 5: **增强其他 Shell 脚本** (entropy.sh, test.sh, search_params.sh)
  - [x] SubTask 5.1: 为 entropy.sh 添加 base adapter 支持
  - [x] SubTask 5.2: 为 test.sh 添加 base adapter 支持
  - [x] SubTask 5.3: 验证 search_params.sh 的实现正确性并根据需要完善
  - [x] SubTask 5.4: 统一所有脚本的配置风格和文档格式

- [x] Task 6: **全面验证修复结果**
  - [x] SubTask 6.1: 验证分支 1（base_adapter + rope_type + adapter）的正确性
  - [x] SubTask 6.2: 验证分支 2（adapter only）的向后兼容性
  - [x] SubTask 6.3: 验证分支 3（base model only）的向后兼容性
  - [x] SubTask 6.4: 测试所有 Shell 脚本的三种模式支持
  - [x] SubTask 6.5: 编写测试用例或验证脚本确认无回归问题

# Task Dependencies
- [Task 2, 3, 4, 5] depend on [Task 1] (必须先修复核心 bug)
- [Task 6] depends on [Task 1, 2, 3, 4, 5] (最后执行全面验证)
