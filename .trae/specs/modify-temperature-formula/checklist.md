# 验证清单

## 温度缩放公式
- [x] 旧的 `_compute_attn_scale` 实现已被注释
- [x] 新的温度缩放公式已实现：`mscale_i(t) = 1 + α * max(0, (ln(max(1, floor(t/b_i))) - ln(L_0)) / ln(L_0))`
- [x] `attn_scale_base` 参数已移除
- [x] 只保留 `attn_scale_coef` 参数

## 约束条件
- [x] 在原始上下文窗口内（`floor(t/b_i) <= L_0`），温度严格等于 1.0
- [x] 在扩展上下文窗口内（`floor(t/b_i) > L_0`），温度按对数增长
- [x] 温度增长速度由 `attn_scale_coef` 控制

## 参数搜索
- [x] `search_attn_scale_params.py` 中已移除 `attn_scale_base` 参数
- [x] `attn_scale_coef` 的默认值为 0.1
- [x] `attn_scale_coef` 的搜索范围为 `[0.05, 0.2]`
- [x] `set_model_attn_scale_params` 函数只设置 `attn_scale_coef`

## 配置文件
- [x] `search_params.sh` 中已移除 `BASE_*` 变量
- [x] `search_params.sh` 中 `COEF_MIN=0.05`、`COEF_MAX=0.2`

## 测试验证
- [x] 在原始上下文窗口内温度严格等于 1.0
- [x] 在扩展上下文窗口内温度按对数增长
- [x] 参数搜索流程运行成功
