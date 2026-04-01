# 验证清单

## 参数接口
- [x] 添加了 `--attn-scale-base-min`、`--attn-scale-base-max`、`--attn-scale-base-steps` 参数
- [x] 参数默认值正确：`base` 范围 `[0.5, 1.0]`，`coef` 范围 `[0.05, 0.3]`
- [x] 参数精度为三位小数

## 约束条件
- [x] 所有生成的参数组合满足 `attn_scale_base <= 1.0`
- [x] 在原始上下文窗口内，温度 `t_base = attn_scale_base <= 1.0`
- [x] 在扩展上下文窗口内，温度 `t_base > attn_scale_base`

## 搜索空间生成
- [x] `generate_grid_search_space` 生成 `(base, coef)` 参数组合
- [x] `generate_random_search_space` 生成 `(base, coef)` 参数组合
- [x] 所有参数值保留三位小数

## 搜索函数
- [x] `run_grid_search` 支持双参数搜索
- [x] `run_random_search` 支持双参数搜索
- [x] `run_bayesian_optimization` 使用二维约束优化
- [x] `run_adaptive_search` 支持双参数自适应搜索
- [x] `run_log_scale_search` 支持双参数搜索

## 搜索策略优化
- [x] 贝叶斯优化使用边界约束 `bounds = [(0.5, 1.0), (0.05, 0.3)]`
- [x] 自适应搜索的搜索中心不超出边界
- [x] 初始采样策略合理（均匀采样或拉丁超立方采样）
- [x] 有约束违反检测和警告

## 配置文件
- [x] `search_params.sh` 包含 `BASE_MIN=0.5`、`BASE_MAX=1.0`、`BASE_STEPS=10`
- [x] `search_params.sh` 包含 `COEF_MIN=0.05`、`COEF_MAX=0.3`、`COEF_STEPS=10`
- [x] python 命令行参数正确
- [x] 打印信息正确

## 测试验证
- [x] 参数组合生成满足约束
- [x] 温度计算逻辑正确
- [x] 单个参数组合评估成功
- [x] 完整搜索流程运行成功
