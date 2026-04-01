# 注意力缩放参数搜索优化规范

## Why
当前实现对 `PerplexityEvaluator` 的使用方式不正确，手动传递长度列表而不是利用其内置的长度生成功能。同时需要评估搜索方案的有效性和参数精度要求。

## What Changes
- **修正评估方式**：使用 `PerplexityEvaluator` 的 `min_length` 和 `max_length` 参数自动生成长度列表，而不是手动传递
- **优化搜索策略**：评估并改进当前的参数搜索方案
- **提高参数精度**：将 `attn_scale_coef` 的精度从两位小数提升到三位小数
- **简化代码逻辑**：移除冗余的多长度评估逻辑，利用 `PerplexityEvaluator.evaluate()` 的内置功能

## Impact
- Affected files:
  - `search_attn_scale_params.py`：主要修改文件
  - `search_params.sh`：参数配置调整
- Affected systems:
  - 参数搜索系统
  - 困惑度评估系统

## ADDED Requirements

### Requirement: 正确使用 PerplexityEvaluator
系统应正确使用 `PerplexityEvaluator` 的内置功能，通过设置 `min_length` 和 `max_length` 自动生成长度列表。

#### Scenario: 自动生成长度列表
- **WHEN** 用户设置 `min_length=4096` 和 `max_length=65536`
- **THEN** `PerplexityEvaluator` 自动生成长度列表 `[4096, 8192, 16384, 32768, 65536]`
- **AND** 对每个长度进行困惑度测试
- **AND** 返回所有长度的困惑度结果

### Requirement: 三位小数精度
系统应支持 `attn_scale_coef` 参数的三位小数精度。

#### Scenario: 参数精度
- **WHEN** 生成参数搜索空间时
- **THEN** 所有 `attn_scale_coef` 值应保留三位小数
- **AND** 范围应为 `[0.050, 0.200]`
- **AND** 步长应足够密集以覆盖关键参数范围

### Requirement: 优化的搜索策略
系统应采用更高效的搜索策略，平衡搜索效率和参数质量。

#### Scenario: 搜索策略评估
- **WHEN** 执行参数搜索时
- **THEN** 应使用自适应搜索或贝叶斯优化
- **AND** 应避免不必要的网格搜索
- **AND** 应支持从之前的结果恢复

### Requirement: 多长度加权评估
系统应对多个长度的困惑度结果进行加权平均，给予更长上下文更高的权重。

#### Scenario: 加权计算
- **WHEN** 获得多个长度的困惑度结果时
- **THEN** 应使用长度加权平均：`weight = length / sum(lengths)`
- **AND** 计算加权困惑度作为最终评估指标
- **AND** 保存每个长度的详细结果

## MODIFIED Requirements

### Requirement: 参数搜索接口
系统应简化参数搜索接口，移除冗余的 `eval_lengths` 参数。

**修改前**：
- 使用 `--eval-lengths "4096,8192,16384,32768,65536"` 手动传递长度列表

**修改后**：
- 使用 `--eval-min-length 4096` 和 `--eval-max-length 65536` 自动生成长度列表
- 利用 `PerplexityEvaluator` 的内置功能

### Requirement: 搜索空间生成
系统应生成三位小数精度的参数值。

**修改前**：
- `attn_scale_coef` 范围：`[0.05, 0.20]`，两位小数
- 步长：16 个采样点

**修改后**：
- `attn_scale_coef` 范围：`[0.050, 0.200]`，三位小数
- 步长：根据搜索方法动态调整（网格搜索 30 个点，自适应搜索动态调整）

## REMOVED Requirements

### Requirement: 手动长度列表
**Reason**：`PerplexityEvaluator` 已有内置的长度列表生成功能，无需手动传递
**Migration**：使用 `min_length` 和 `max_length` 参数替代

### Requirement: 冗余的多长度评估逻辑
**Reason**：`PerplexityEvaluator.evaluate()` 已经返回所有长度的结果，无需在 `evaluate_params` 中重复实现
**Migration**：直接使用 `PerplexityEvaluator` 的返回结果
