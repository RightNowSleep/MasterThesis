# 统一 entropy.py 文件命名规范

## Why
当前 `entropy.py` 的文件命名函数 `generate_save_filename` 仅基于 RoPE 类型生成文件名，无法区分不同的 adapter。当评估 adapter 时，文件名会是 `llama-7b_none.json`，无法标识具体的 adapter，导致结果文件混淆。需要使其与其他测试方法（如 perplexity.py、performance.py）保持一致的命名规范，并支持 adapter 标识。

## What Changes
- 修改 `generate_save_filename` 函数，添加 adapter 路径处理逻辑
- 当使用 adapter 时，从 adapter 路径中提取标识符并加入文件名
- 保持与 perplexity.py、performance.py 一致的命名风格
- 更新函数文档和示例

## Impact
- Affected specs: 熵评估结果文件命名
- Affected code: 
  - `/home/linzhen/workspace/MasterThesis/eval/entropy.py` (主要修改)

## ADDED Requirements

### Requirement: Adapter 文件命名支持
系统 SHALL 在评估 adapter 时生成包含 adapter 标识符的文件名，以便区分不同的 adapter 评估结果。

#### Scenario: Adapter 文件命名
- **WHEN** 用户使用 `--adapter-path finetunes/continued_pretrain/dual-rope_20260402_113443` 参数
- **THEN** 文件名应包含 adapter 标识符，如 `llama-7b_adapter_dual-rope_20260402_113443.json`
- **AND** 文件名应清晰标识这是 adapter 评估结果

#### Scenario: RoPE 方法文件命名
- **WHEN** 用户使用 `--rope-type linear --rope-dynamic` 参数
- **THEN** 文件名应为 `llama-7b_linear_dynamic.json`
- **AND** 与当前命名方式保持一致

#### Scenario: Adapter + RoPE 组合文件命名
- **WHEN** 用户同时使用 adapter 和 RoPE 方法
- **THEN** 文件名应同时包含 adapter 标识符和 RoPE 信息
- **AND** 格式应为 `{model}_adapter_{adapter_name}_{rope_type}_{rope_config}.json`

### Requirement: 命名规范一致性
系统 SHALL 使用与其他测试方法一致的文件命名规范，确保所有评估脚本的输出文件命名风格统一。

#### Scenario: 与 perplexity.py 一致
- **WHEN** 比较不同评估脚本的文件命名
- **THEN** 应使用相同的命名模式和逻辑
- **AND** 文件名格式应保持一致

## MODIFIED Requirements

### Requirement: 文件命名函数
系统 SHALL 更新 `generate_save_filename` 函数以支持 adapter 路径处理。

#### Scenario: 函数参数更新
- **WHEN** 调用 `generate_save_filename` 函数
- **THEN** 函数应接受 args 对象，包含 `adapter_path` 属性
- **AND** 函数应正确处理 adapter 路径并生成合适的文件名

## REMOVED Requirements
无
