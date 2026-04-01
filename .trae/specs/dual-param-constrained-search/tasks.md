# Tasks

## Task 1: 恢复双参数搜索接口
- [x] Task 1.1: 添加 `--attn-scale-base-min`、`--attn-scale-base-max`、`--attn-scale-base-steps` 命令行参数
- [x] Task 1.2: 设置合理的默认值：`base` 范围 `[0.5, 1.0]`，步长 10；`coef` 范围 `[0.05, 0.3]`，步长 10
- [x] Task 1.3: 更新 `main()` 函数的参数打印逻辑

## Task 2: 修改搜索空间生成函数
- [x] Task 2.1: 修改 `generate_grid_search_space` 函数，生成 `(base, coef)` 参数组合
- [x] Task 2.2: 修改 `generate_random_search_space` 函数，生成 `(base, coef)` 参数组合
- [x] Task 2.3: 确保所有参数值保留三位小数
- [x] Task 2.4: 添加约束验证：确保 `attn_scale_base <= 1.0`

## Task 3: 修改 evaluate_params 函数
- [x] Task 3.1: 恢复 `attn_scale_base` 参数
- [x] Task 3.2: 调用 `set_model_attn_scale_params(model, attn_scale_base, attn_scale_coef)`
- [x] Task 3.3: 保持多长度评估和加权平均逻辑

## Task 4: 更新所有搜索函数
- [x] Task 4.1: 更新 `run_grid_search` 函数，支持双参数搜索
- [x] Task 4.2: 更新 `run_random_search` 函数，支持双参数搜索
- [x] Task 4.3: 更新 `run_bayesian_optimization` 函数，使用二维约束优化
- [x] Task 4.4: 更新 `run_adaptive_search` 函数，支持双参数自适应搜索
- [x] Task 4.5: 更新 `run_log_scale_search` 函数，支持双参数搜索

## Task 5: 优化搜索策略
- [x] Task 5.1: 为贝叶斯优化添加边界约束：`bounds = [(0.5, 1.0), (0.05, 0.3)]`
- [x] Task 5.2: 为自适应搜索添加约束检查，确保搜索中心不超出边界
- [x] Task 5.3: 优化初始采样策略，使用拉丁超立方采样或均匀采样
- [x] Task 5.4: 添加约束违反检测和警告

## Task 6: 更新配置文件
- [x] Task 6.1: 更新 `search_params.sh`，添加 `BASE_MIN=0.5`、`BASE_MAX=1.0`、`BASE_STEPS=10`
- [x] Task 6.2: 更新 `COEF_MIN=0.05`、`COEF_MAX=0.3`、`COEF_STEPS=10`
- [x] Task 6.3: 更新 python 命令行参数
- [x] Task 6.4: 更新打印信息

## Task 7: 测试和验证
- [x] Task 7.1: 验证参数组合生成是否满足约束
- [x] Task 7.2: 验证温度计算逻辑是否正确
- [x] Task 7.3: 测试单个参数组合的评估流程
- [x] Task 7.4: 测试完整的搜索流程

# Task Dependencies
- [Task 2] 依赖 [Task 1]（搜索空间生成需要参数接口）
- [Task 3] 依赖 [Task 1]（评估函数需要参数接口）
- [Task 4] 依赖 [Task 1]、[Task 2]、[Task 3]（搜索函数需要完整的参数支持）
- [Task 5] 依赖 [Task 4]（搜索策略优化需要搜索函数完成）
- [Task 6] 依赖 [Task 1]（配置文件需要参数接口）
- [Task 7] 依赖 [Task 1] 到 [Task 6]（测试需要所有功能完成）
