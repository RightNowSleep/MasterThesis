# Tasks

- [x] Task 1: 修复 pe_llama.py 中的维度计算
  - [x] SubTask 1.1: 检查并修复 inv_freq 的维度计算逻辑
  - [x] SubTask 1.2: 确保 cos/sin 的最终维度正确

- [x] Task 2: 更新 model_loader.py
  - [x] SubTask 2.1: 在 `_ROPE_TYPES_WITH_DYNAMIC_FLAG` 中添加 `dual-rope` 和 `dual-rope-scaled`
  - [x] SubTask 2.2: 更新帮助文档

- [x] Task 3: 更新 modeling_llama.py
  - [x] SubTask 3.1: 添加 `dual-rope` 类型的初始化逻辑
  - [x] SubTask 3.2: 添加 `dual-rope-scaled` 类型的初始化逻辑
  - [x] SubTask 3.3: 更新错误提示信息

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 1]
