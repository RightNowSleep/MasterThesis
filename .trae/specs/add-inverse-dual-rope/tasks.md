# Tasks

- [x] Task 1: 在 pe_llama.py 中实现 LlamaInverseDualRoPEEmbedding 类
  - [x] 1.1 实现基础 Inverse-Dual-RoPE 嵌入类
    - 继承 nn.Module
    - 实现 __init__: 初始化参数，计算 i_star，分割 inv_freq 为 inv_freq_1 (高频) 和 inv_freq_2 (低频)
    - 实现 _set_cos_sin_cache: 高频用 t // L_0，低频用 t % L_0
    - 实现 forward: 支持动态和静态模式
  - [x] 1.2 实现 LlamaInverseDualRoPEScaledEmbedding 类
    - 继承 LlamaInverseDualRoPEEmbedding
    - 添加 _compute_attn_scale 方法（使用 log_{L_0}(t) 公式）
    - 重写 _set_cos_sin_cache 以应用温度缩放

- [x] Task 2: 在 configuration_llama.py 中添加新类型支持
  - [x] 2.1 在 valid_types 列表中添加 "inverse-dual-rope" 和 "inverse-dual-rope-scaled"
  - [x] 2.2 更新文档字符串中的类型列表

- [x] Task 3: 在 model_loader.py 中注册新类型
  - [x] 3.1 在 _ROPE_TYPES_WITH_DYNAMIC_FLAG 集合中添加新类型
  - [x] 3.2 在 add_args_model 的 help 文本中添加新类型说明

- [x] Task 4: 在 modeling_llama.py 中集成新 RoPE 类型
  - [x] 4.1 添加 "inverse-dual-rope" 路由分支 → LlamaInverseDualRoPEEmbedding
  - [x] 4.2 添加 "inverse-dual-rope-scaled" 路由分支 → LlamaInverseDualRoPEScaledEmbedding
  - [x] 4.3 更新错误消息中的有效类型列表

- [x] Task 5: 在 continued_pretrain.py 中实现渐进式长度训练功能
  - [x] 5.1 添加 --progressive-length 参数到 argument parser
  - [x] 5.2 实现渐进式长度序列生成函数 (generate_progressive_lengths)
  - [x] 5.3 修改训练循环以支持多阶段长度训练
  - [x] 5.4 实现阶段间的平滑过渡逻辑
  - [x] 5.5 更新日志输出以显示当前训练阶段信息

- [x] Task 6: 在 continued_pretrain.sh 中添加渐进式训练配置
  - [x] 6.1 添加 PROGRESSIVE_LENGTH 配置变量
  - [x] 6.2 将 --progressive-length 参数传递给训练脚本

- [x] Task 7: 测试验证
  - [x] 7.1 验证 Inverse-Dual-RoPE 正确初始化和前向传播
  - [x] 7.2 验证 Scaled 版本的温度缩放正确性
  - [x] 7.3 验证渐进式长度训练流程的正确性

# Task Dependencies
- [Task 2, Task 3] 可以与 Task 1 并行执行
- [Task 4] 依赖 [Task 1]
- [Task 5, Task 6] 可以并行执行，但都依赖配置完成
- [Task 7] 依赖所有其他任务完成
