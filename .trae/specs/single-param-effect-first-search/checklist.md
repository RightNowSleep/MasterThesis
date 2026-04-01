# 验证清单

## 参数接口
- [x] 移除了 `--attn-scale-base-min`、`--attn-scale-base-max`、`--attn-scale-base-steps` 参数
- [x] `attn_scale_base` 固定为 1.0
- [x] `attn_scale_coef` 范围为 `[0.05, 0.3]`
- [x] 参数精度为三位小数

## 约束条件
- [x] 在原始上下文窗口内，温度 `t_base = 1.0`（严格等于）
- [x] 在扩展上下文窗口内，温度 `t_base = 1.0 + attn_scale_coef * log(s) > 1.0`
- [x] `attn_scale_base` 固定为 1.0，不参与搜索

## 搜索空间生成
- [x] `generate_grid_search_space` 只生成 `attn_scale_coef` 的参数值
- [x] `generate_random_search_space` 只生成 `attn_scale_coef` 的参数值
- [x] 所有参数值保留三位小数
- [x] 网格搜索至少有 50 个采样点

## 搜索函数
- [x] `run_grid_search` 支持单参数搜索，使用密集采样（50 个点）
- [x] `run_random_search` 支持单参数搜索
- [x] `run_bayesian_optimization` 使用一维优化，迭代次数至少 50 次
- [x] `run_adaptive_search` 使用 4-5 个阶段
- [x] `run_log_scale_search` 支持单参数搜索

## 搜索策略优化（效果优先）
- [x] 网格搜索默认使用 50 个采样点
- [x] 自适应搜索使用 4-5 个阶段
- [x] 贝叶斯优化使用 50-100 次迭代
- [x] 有详细的搜索过程日志

## 配置文件
- [x] `search_params.sh` 移除了 `BASE_*` 变量
- [x] `search_params.sh` 包含 `COEF_MIN=0.05`、`COEF_MAX=0.3`、`COEF_STEPS=50`
- [x] python 命令行参数正确
- [x] 打印信息显示 `base=1.0 (fixed)`

## 测试验证
- [x] 在原始上下文窗口内温度严格等于 1.0
- [x] 参数搜索只搜索 `attn_scale_coef`
- [x] 单个参数组合评估成功
- [x] 完整搜索流程运行成功，找到最优参数
