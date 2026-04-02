# Tasks

- [x] Task 1: 分析现有文件命名逻辑
  - [x] SubTask 1.1: 读取 entropy.py 的 generate_save_filename 函数
  - [x] SubTask 1.2: 读取 perplexity.py 和 performance.py 的 generate_save_filename 函数
  - [x] SubTask 1.3: 分析 args 对象中的 adapter_path 属性

- [x] Task 2: 设计 adapter 文件命名方案
  - [x] SubTask 2.1: 确定从 adapter 路径提取标识符的方法
  - [x] SubTask 2.2: 设计文件名格式（adapter-only、adapter+rope）
  - [x] SubTask 2.3: 确保与现有命名规范兼容

- [x] Task 3: 实现文件命名逻辑修改
  - [x] SubTask 3.1: 修改 generate_save_filename 函数
  - [x] SubTask 3.2: 添加 adapter 路径处理逻辑
  - [x] SubTask 3.3: 更新函数文档和示例

- [x] Task 4: 测试和验证
  - [x] SubTask 4.1: 验证 RoPE-only 模式文件命名正确
  - [x] SubTask 4.2: 验证 Adapter-only 模式文件命名正确
  - [x] SubTask 4.3: 验证 Adapter+RoPE 组合模式文件命名正确

# Task Dependencies
- [Task 2] depends on [Task 1] - 需要先分析现有逻辑
- [Task 3] depends on [Task 2] - 需要先设计命名方案
- [Task 4] depends on [Task 3] - 需要先完成实现
