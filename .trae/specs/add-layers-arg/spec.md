# 图3指定绘制层参数 Spec

## Why
当前图3（`fig03_entropy_vs_position`）的层选择完全由 `_auto_select_layers` 自动决定（选择 head-entropy 标准差最高的 8 层），用户无法指定想要观察的特定层。需要支持通过命令行参数指定绘制层，以便对特定层进行针对性分析。

## What Changes
- 修改 `_auto_select_layers` 函数，增加 `preferred_layers` 参数，支持用户指定优先层
- 修改 `plot_entropy_vs_position` 函数，增加 `preferred_layers` 参数，传递给 `_auto_select_layers`
- 修改 `plot_all` 函数，增加 `preferred_layers` 参数，传递给图3绘制函数
- 修改 `_build_parser`，增加 `--layers` 命令行参数
- 修改 `main` 函数，解析 `--layers` 并传递到调用链中
- 修改 `entropy.sh`，增加 `PLOT_LAYERS` 配置变量，在 Part 2 调用时传递 `--layers`

## Impact
- Affected code: `eval/plot_entropy.py`（`_auto_select_layers`、`plot_entropy_vs_position`、`plot_all`、`_build_parser`、`main`）
- Affected code: `entropy.sh`（Part 2 的 python 调用命令）
- 向后兼容：不指定 `--layers` 时行为与原来完全一致

## ADDED Requirements

### Requirement: 图3支持指定绘制层
系统 SHALL 提供 `--layers` 命令行参数，允许用户指定图3要绘制的层索引列表。

#### Scenario: 未指定 --layers
- **WHEN** 用户不提供 `--layers` 参数
- **THEN** 按原规则自动选择 8 层（head-entropy 标准差最高的 8 层），行为与修改前完全一致

#### Scenario: 指定少于8层
- **WHEN** 用户通过 `--layers` 指定了 X 个层（X < 8）
- **THEN** 优先使用用户指定的层，剩余 (8-X) 个位置由原规则（head-entropy 标准差降序）自动补充，且不与已选层重复

#### Scenario: 指定恰好8层
- **WHEN** 用户通过 `--layers` 指定了恰好 8 个层
- **THEN** 直接使用用户指定的 8 层，不做补充或裁剪

#### Scenario: 指定超过8层
- **WHEN** 用户通过 `--layers` 指定了 X 个层（X > 8）
- **THEN** 截取前 8 层，丢弃超出部分

#### Scenario: 指定层索引越界
- **WHEN** 用户指定的层索引超出模型层数范围
- **THEN** 静默忽略越界索引，仅保留有效索引

#### Scenario: 指定层索引含重复
- **WHEN** 用户指定的层索引列表中包含重复值
- **THEN** 去重后使用，保持首次出现的顺序

### Requirement: entropy.sh 支持传递层参数
系统 SHALL 在 `entropy.sh` 中增加 `PLOT_LAYERS` 配置变量，在 Part 2 调用 `plot_entropy.py` 时将其作为 `--layers` 参数传递。

#### Scenario: PLOT_LAYERS 为空
- **WHEN** `PLOT_LAYERS` 变量为空字符串
- **THEN** 不传递 `--layers` 参数，使用默认自动选择行为

#### Scenario: PLOT_LAYERS 指定了层
- **WHEN** `PLOT_LAYERS` 变量设为 `"31 13 8 23 30 9 28 14"`
- **THEN** 在调用 `python plot_entropy.py` 时追加 `--layers 31 13 8 23 30 9 28 14`
