# 增强 entropy.sh 脚本支持 Adapter 评估

## Why
当前 `entropy.sh` 脚本仅支持 RoPE 方法的熵评估，无法评估 fine-tuned adapter 的表现。需要使其与 `eval.sh` 保持一致的配置方式，支持 RoPE 方法和 Adapter 两种加载模式，以便进行全面的熵分析对比。

## What Changes
- 添加 ADAPTER 模式支持，允许评估 fine-tuned adapters
- 添加 ROPE 和 ADAPTER 标志控制，灵活选择评估模式
- 重构 ROPE_METHODS 配置格式，与 eval.sh 保持一致
- 添加 ADAPTER_PATHS 配置数组
- 构建 METHODS 统一列表，统一处理 RoPE 和 Adapter 评估
- 修改 entropy.py 调用方式，支持 adapter 参数传递

## Impact
- Affected specs: 熵评估流程
- Affected code: 
  - `/home/linzhen/workspace/MasterThesis/entropy.sh` (主要修改)
  - `/home/linzhen/workspace/MasterThesis/eval/entropy.py` (可能需要确认是否支持 adapter 参数)

## ADDED Requirements

### Requirement: Adapter 模式支持
系统 SHALL 提供对 fine-tuned adapter 的熵评估支持，允许用户通过配置 adapter 路径来评估 adapter 模型。

#### Scenario: Adapter 熵评估
- **WHEN** 用户设置 ADAPTER=true 并配置 ADAPTER_PATHS
- **THEN** 系统应加载指定的 adapter 并执行熵评估
- **AND** 结果应保存到指定的输出目录

### Requirement: 双模式控制
系统 SHALL 提供 ROPE 和 ADAPTER 两个布尔标志，允许用户独立控制是否评估 RoPE 方法或 Adapter 模型。

#### Scenario: 仅评估 RoPE 方法
- **WHEN** 用户设置 ROPE=true 和 ADAPTER=false
- **THEN** 系统应仅评估 ROPE_METHODS 中配置的方法

#### Scenario: 仅评估 Adapter
- **WHEN** 用户设置 ROPE=false 和 ADAPTER=true
- **THEN** 系统应仅评估 ADAPTER_PATHS 中配置的 adapter

#### Scenario: 同时评估两者
- **WHEN** 用户设置 ROPE=true 和 ADAPTER=true
- **THEN** 系统应评估所有配置的 RoPE 方法和 Adapter

### Requirement: 配置格式统一
系统 SHALL 使用与 eval.sh 一致的配置格式，确保两个脚本的配置方式保持一致。

#### Scenario: ROPE_METHODS 格式
- **WHEN** 用户配置 ROPE_METHODS
- **THEN** 每个方法应为完整的参数字符串格式，如 `"--rope-type dual-rope --rope-dynamic"`

#### Scenario: ADAPTER_PATHS 格式
- **WHEN** 用户配置 ADAPTER_PATHS
- **THEN** 每个 adapter 应为完整的参数字符串格式，如 `"--adapter-path ${ADAPTER_DIR}/method_name"`

## MODIFIED Requirements

### Requirement: 熵评估流程
系统 SHALL 支持统一的评估流程，能够处理 RoPE 方法和 Adapter 两种模型加载方式。

#### Scenario: 统一的评估循环
- **WHEN** 系统执行熵评估
- **THEN** 应遍历 METHODS 列表中的所有配置
- **AND** 每个配置应正确传递给 entropy.py 脚本

## REMOVED Requirements
无
