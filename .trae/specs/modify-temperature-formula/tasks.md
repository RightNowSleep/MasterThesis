# Tasks

## Task 1: 修改 pe_llama.py 中的温度缩放公式
- [x] Task 1.1: 注释旧的 `_compute_attn_scale` 实现
- [x] Task 1.2: 实现新的温度缩放公式：`mscale_i(t) = 1 + α * max(0, (ln(max(1, floor(t/b_i))) - ln(L_0)) / ln(L_0))`
- [x] Task 1.3: 移除 `attn_scale_base` 参数，只保留 `attn_scale_coef`
- [x] Task 1.4: 更新 `__init__` 方法，移除 `attn_scale_base` 参数

## Task 2: 更新参数搜索脚本
- [x] Task 2.1: 确认 `search_attn_scale_params.py` 中已经移除了 `attn_scale_base` 参数
- [x] Task 2.2: 更新 `attn_scale_coef` 的默认值和搜索范围：`[0.05, 0.2]`
- [x] Task 2.3: 更新 `set_model_attn_scale_params` 函数，只设置 `attn_scale_coef`

## Task 3: 更新配置文件
- [x] Task 3.1: 更新 `search_params.sh`，设置 `COEF_MIN=0.05`、`COEF_MAX=0.2`
- [x] Task 3.2: 确认配置文件中已经移除了 `BASE_*` 变量

## Task 4: 测试和验证
- [x] Task 4.1: 验证在原始上下文窗口内温度严格等于 1.0
- [x] Task 4.2: 验证在扩展上下文窗口内温度按对数增长
- [x] Task 4.3: 测试参数搜索流程

# Task Dependencies
- [Task 2] 依赖 [Task 1]（参数搜索脚本需要新的参数接口）
- [Task 3] 依赖 [Task 1] 和 [Task 2]（配置文件需要参数接口）
- [Task 4] 依赖 [Task 1] 到 [Task 3]（测试需要所有功能完成）
