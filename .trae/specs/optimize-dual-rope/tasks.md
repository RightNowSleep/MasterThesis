# Tasks

- [x] Task 1: 优化 LlamaDualRoPEEmbedding 类
  - [x] SubTask 1.1: 在 `__init__` 中添加 `inv_freq_1` 和 `inv_freq_2` 缓冲区的预计算
  - [x] SubTask 1.2: 修改 `_set_cos_sin_cache` 方法使用预计算的 inv_freq 部分
  - [x] SubTask 1.3: 修改 `forward` 方法使用预计算的 inv_freq 部分
  - [x] SubTask 1.4: 验证优化后的功能正确性

- [x] Task 2: 优化 LlamaDualRoPEScaledEmbedding 类
  - [x] SubTask 2.1: 修改 `_set_cos_sin_cache` 方法避免重复计算位置信息
  - [x] SubTask 2.2: 优化 attention scale 的计算逻辑
  - [x] SubTask 2.3: 修改 `forward` 方法优化 attention scale 的应用
  - [x] SubTask 2.4: 验证优化后的功能正确性

- [x] Task 3: 性能测试和验证
  - [x] SubTask 3.1: 编写性能测试脚本对比优化前后的速度
  - [x] SubTask 3.2: 验证优化后的数值结果与优化前一致
  - [x] SubTask 3.3: 测试静态模式和动态模式的正确性

# Task Dependencies
- Task 2 依赖 Task 1（因为 LlamaDualRoPEScaledEmbedding 继承自 LlamaDualRoPEEmbedding）
- Task 3 依赖 Task 1 和 Task 2
