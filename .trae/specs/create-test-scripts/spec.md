# 创建测试脚本 Spec

## Why
项目中有多种RoPE方法实现（none、linear、ntk、part-ntk、yarn、my-rope、dynamic-my-rope），需要创建命令行脚本来方便地对这些方法进行各种测试。目前缺少统一的测试脚本入口。

## What Changes
- 创建 `test.sh` - 调用 `test.py` 的统一测试脚本，支持通过子命令选择测试类型
- 创建 `eval.sh` - 调用 `eval/` 文件夹下具体测试文件的脚本

## Impact
- Affected specs: 无
- Affected code: 新增 `test.sh` 和 `eval.sh` 两个脚本文件

## ADDED Requirements

### Requirement: test.sh 脚本
系统应提供 `test.sh` 脚本，用于调用 `test.py` 进行RoPE方法测试。

#### Scenario: 执行困惑度测试
- **WHEN** 用户运行 `./test.sh perplexity --rope-type linear --rope-factor 4.0`
- **THEN** 脚本调用 `test.py perplexity` 并传递相应参数

#### Scenario: 执行密钥检索测试
- **WHEN** 用户运行 `./test.sh passkey --rope-type ntk --rope-factor 8.0`
- **THEN** 脚本调用 `test.py passkey` 并传递相应参数

#### Scenario: 执行质量测试
- **WHEN** 用户运行 `./test.sh quality --rope-type yarn --rope-factor 4.0`
- **THEN** 脚本调用 `test.py quality` 并传递相应参数

#### Scenario: 执行性能测试
- **WHEN** 用户运行 `./test.sh performance --rope-type dynamic-my-rope`
- **THEN** 脚本调用 `test.py performance` 并传递相应参数

### Requirement: eval.sh 脚本
系统应提供 `eval.sh` 脚本，用于调用 `eval/` 文件夹下的具体测试文件。

#### Scenario: 执行困惑度评估
- **WHEN** 用户运行 `./eval.sh perplexity --rope-type linear --rope-factor 4.0`
- **THEN** 脚本调用 `eval/perplexity.py` 并传递相应参数

#### Scenario: 执行密钥检索评估
- **WHEN** 用户运行 `./eval.sh passkey --rope-type ntk --rope-factor 8.0`
- **THEN** 脚本调用 `eval/passkey.py` 并传递相应参数

#### Scenario: 执行质量评估
- **WHEN** 用户运行 `./eval.sh quality --rope-type yarn --rope-factor 4.0`
- **THEN** 脚本调用 `eval/quality.py` 并传递相应参数

#### Scenario: 执行性能评估
- **WHEN** 用户运行 `./eval.sh performance --rope-type dynamic-my-rope`
- **THEN** 脚本调用 `eval/performance.py` 并传递相应参数

### Requirement: RoPE类型支持
两个脚本都应支持所有已实现的RoPE类型：
- `none` - 标准RoPE，无缩放
- `linear` - Position Interpolation (PI)
- `ntk` - NTK-aware scaling
- `part-ntk` - NTK-by-parts scaling
- `yarn` - YaRN
- `my-rope` - 自定义静态RoPE
- `dynamic-my-rope` - 自定义动态RoPE

### Requirement: 通用参数支持
两个脚本都应支持以下通用参数：
- `--model-name` - 模型名称或路径
- `--rope-type` - RoPE类型
- `--rope-factor` - 扩展因子
- `--rope-dynamic` - 是否启用动态缩放（适用于linear/ntk/part-ntk/yarn）
- `--max-length` - 最大序列长度
- `--min-length` - 最小序列长度
- `--device` - 设备选择
- `--dtype` - 数据类型
