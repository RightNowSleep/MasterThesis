# Tasks

- [x] Task 1: 分析 entropy.py 是否支持 adapter 参数
  - [x] SubTask 1.1: 读取 eval/entropy.py 源码
  - [x] SubTask 1.2: 确认是否已支持 --adapter-path 参数
  - [x] SubTask 1.3: 如不支持，记录需要添加的功能

- [x] Task 2: 重构 entropy.sh 脚本配置部分
  - [x] SubTask 2.1: 添加 ROPE 和 ADAPTER 模式标志
  - [x] SubTask 2.2: 修改 ROPE_METHODS 格式为完整参数字符串
  - [x] SubTask 2.3: 添加 ADAPTER_DIR 和 ADAPTER_PATHS 配置
  - [x] SubTask 2.4: 添加 METHODS 数组构建逻辑

- [x] Task 3: 修改熵评估循环逻辑
  - [x] SubTask 3.1: 更新 Part 1 的循环，使用 METHODS 数组
  - [x] SubTask 3.2: 确保参数正确传递给 entropy.py

- [x] Task 4: 测试和验证
  - [x] SubTask 4.1: 验证 ROPE 模式正常工作
  - [x] SubTask 4.2: 验证 ADAPTER 模式正常工作
  - [x] SubTask 4.3: 验证混合模式正常工作

# Task Dependencies
- [Task 2] depends on [Task 1] - 需要先确认 entropy.py 的参数支持情况
- [Task 3] depends on [Task 2] - 需要先完成配置部分的重构
- [Task 4] depends on [Task 3] - 需要先完成脚本修改
