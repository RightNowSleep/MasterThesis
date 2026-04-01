# Tasks

## Task 1: 恢复单参数搜索接口
- [x] Task 1.1: 移除 `--attn-scale-base-min`、`--attn-scale-base-max`、`--attn-scale-base-steps` 参数
- [x] Task 1.2: 更新 `--attn-scale-coef-*` 参数默认值：范围 `[0.05, 0.3]`，步长 50
- [x] Task 1.3: 更新 `main()` 函数的参数打印逻辑，显示 `attn_scale_base=1.0 (fixed)`

## Task 2: 修改搜索空间生成函数
- [x] Task 2.1: 修改 `generate_grid_search_space` 函数，只生成 `attn_scale_coef` 的参数值
- [x] Task 2.2: 修改 `generate_random_search_space` 函数，只生成 `attn_scale_coef` 的参数值
- [x] Task 2.3: 确保参数值保留三位小数
- [x] Task 2.4: 增加采样密度：网格搜索至少 50 个点

## Task 3: 修改 evaluate_params 函数
- [x] Task 3.1: 移除 `attn_scale_base` 参数，固定为 1.0
- [x] Task 3.2: 调用 `set_model_attn_scale_params(model, 1.0, attn_scale_coef)`
- [x] Task 3.3: 保持多长度评估和加权平均逻辑

## Task 4: 更新所有搜索函数
- [x] Task 4.1: 更新 `run_grid_search` 函数，支持单参数搜索，使用密集采样
- [x] Task 4.2: 更新 `run_random_search` 函数，支持单参数搜索
- [x] Task 4.3: 更新 `run_bayesian_optimization` 函数，使用一维优化，增加迭代次数到 50
- [x] Task 4.4: 更新 `run_adaptive_search` 函数，增加阶段数到 4-5 个
- [x] Task 4.5: 更新 `run_log_scale_search` 函数，支持单参数搜索

## Task 5: 优化搜索策略（效果优先）
- [x] Task 5.1: 网格搜索默认使用 50 个采样点
- [x] Task 5.2: 自适应搜索使用 4-5 个阶段，每个阶段有足够的采样点
- [x] Task 5.3: 贝叶斯优化使用 50-100 次迭代
- [x] Task 5.4: 添加详细的搜索过程日志，记录每个参数的评估结果

## Task 6: 更新配置文件
- [x] Task 6.1: 更新 `search_params.sh`，移除 `BASE_*` 变量
- [x] Task 6.2: 更新 `COEF_MIN=0.05`、`COEF_MAX=0.3`、`COEF_STEPS=50`
- [x] Task 6.3: 更新 python 命令行参数，移除 base 相关参数
- [x] Task 6.4: 更新打印信息，显示 `base=1.0 (fixed)`

## Task 7: 测试和验证
- [x] Task 7.1: 验证在原始上下文窗口内温度严格等于 1.0
- [x] Task 7.2: 验证参数搜索只搜索 `attn_scale_coef`
- [x] Task 7.3: 测试单个参数组合的评估流程
- [x] Task 7.4: 测试完整的搜索流程，确保找到最优参数

# Task Dependencies
- [Task 2] 依赖 [Task 1]（搜索空间生成需要参数接口）
- [Task 3] 依赖 [Task 1]（评估函数需要参数接口）
- [Task 4] 依赖 [Task 1]、[Task 2]、[Task 3]（搜索函数需要完整的参数支持）
- [Task 5] 依赖 [Task 4]（搜索策略优化需要搜索函数完成）
- [Task 6] 依赖 [Task 1]（配置文件需要参数接口）
- [Task 7] 依赖 [Task 1] 到 [Task 6]（测试需要所有功能完成）
