# 优化 search_params.sh 直接运行配置 Spec

## Why
用户希望能够直接通过 `bash search_params.sh` 或 `./search_params.sh` 运行脚本，使用最优的搜索配置，无需手动修改参数。

## What Changes
- 将 `SEARCH_METHOD` 从 `adaptive` 改为 `bohb`（BOHB 是经过研究验证的最优搜索策略）
- 确保 BOHB 相关参数配置为最优值
- 更新脚本注释，反映新的温度缩放公式
- 确保脚本可直接执行

## Impact
- Affected code: `search_params.sh`

## ADDED Requirements

### Requirement: 直接运行最优配置
系统应当允许用户通过 `bash search_params.sh` 或 `./search_params.sh` 直接运行最优的参数搜索配置。

#### Scenario: 用户直接运行脚本
- **WHEN** 用户执行 `bash search_params.sh` 或 `./search_params.sh`
- **THEN** 脚本使用 BOHB 搜索方法，参数范围为 [0.01, 0.3]，初始采样 25 个点，迭代 100 次

### Requirement: BOHB 搜索配置
脚本应当使用 BOHB (Bayesian Optimization + HyperBand) 作为默认搜索方法。

#### Scenario: BOHB 配置验证
- **WHEN** 脚本运行时
- **THEN** 使用以下最优配置：
  - `SEARCH_METHOD="bohb"`
  - `COEF_MIN=0.01`
  - `COEF_MAX=0.3`
  - `BOHB_INITIAL_SAMPLES=25`
  - `BOHB_ITERATIONS=100`
  - `BOHB_EARLY_STOP_FACTOR=3`

## MODIFIED Requirements

### Requirement: 脚本注释更新
脚本顶部的注释应当反映新的温度缩放公式。

**原注释**:
```
t_base = attn_scale_base + attn_scale_coef * torch.log(s)
```

**新注释**:
```
mscale_i(t) = 1 + α * max(0, (ln(max(1, floor(t/b_i))) - ln(L_0)) / ln(L_0))
```
