# Tasks

## Task 1: 修正 evaluate_params 函数，正确使用 PerplexityEvaluator
- [x] Task 1.1: 移除 `eval_lengths` 参数，改为 `eval_min_length` 和 `eval_max_length`
- [x] Task 1.2: 使用 `PerplexityEvaluator(min_length=eval_min_length, max_length=eval_max_length)` 自动生成长度列表
- [x] Task 1.3: 从 `PerplexityEvaluator.evaluate()` 的返回结果中提取多个长度的困惑度
- [x] Task 1.4: 实现长度加权平均计算：`weight = length / sum(lengths)`

## Task 2: 提升参数精度到三位小数
- [x] Task 2.1: 修改 `generate_grid_search_space` 函数，使用 `round(v, 3)` 保留三位小数
- [x] Task 2.2: 修改 `generate_random_search_space` 函数，使用 `round(v, 3)` 保留三位小数
- [x] Task 2.3: 更新命令行参数默认值：`attn_scale_coef_min=0.050`, `attn_scale_coef_max=0.200`
- [x] Task 2.4: 增加网格搜索步长到 30 个点，提高搜索密度

## Task 3: 优化搜索策略
- [x] Task 3.1: 评估当前搜索方法的有效性
- [x] Task 3.2: 优化自适应搜索的初始参数和收敛条件
- [x] Task 3.3: 改进贝叶斯优化的初始采样策略
- [x] Task 3.4: 添加搜索过程的详细日志记录

## Task 4: 更新命令行参数和配置文件
- [x] Task 4.1: 修改 `parse_args()` 函数，移除 `--eval-lengths`，添加 `--eval-min-length` 和 `--eval-max-length`
- [x] Task 4.2: 更新 `search_params.sh`，使用新的参数格式
- [x] Task 4.3: 更新 `main()` 函数中的参数解析和打印逻辑
- [x] Task 4.4: 更新结果保存逻辑，保存每个长度的详细困惑度结果

## Task 5: 更新所有搜索函数
- [x] Task 5.1: 更新 `run_grid_search` 函数，使用新的参数接口
- [x] Task 5.2: 更新 `run_random_search` 函数，使用新的参数接口
- [x] Task 5.3: 更新 `run_bayesian_optimization` 函数，使用新的参数接口
- [x] Task 5.4: 更新 `run_adaptive_search` 函数，使用新的参数接口
- [x] Task 5.5: 更新 `run_log_scale_search` 函数，使用新的参数接口

## Task 6: 测试和验证
- [x] Task 6.1: 测试单个参数组合的评估流程
- [x] Task 6.2: 验证长度列表生成是否正确
- [x] Task 6.3: 验证加权平均计算是否正确
- [x] Task 6.4: 测试完整的搜索流程

# Task Dependencies
- [Task 2] 依赖 [Task 1]（参数精度提升需要先修正评估函数）
- [Task 3] 依赖 [Task 1]（搜索策略优化需要正确的评估函数）
- [Task 4] 依赖 [Task 1] 和 [Task 2]（参数更新需要先完成函数修改）
- [Task 5] 依赖 [Task 1]、[Task 2] 和 [Task 4]（搜索函数更新需要新的参数接口）
- [Task 6] 依赖 [Task 1] 到 [Task 5]（测试需要所有功能完成）
