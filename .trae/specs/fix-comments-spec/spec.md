# 代码注释规范修正 Spec

## Why
当前项目中存在大量.py和.sh文件的注释问题，包括：注释与代码逻辑不匹配、注释格式不符合行业规范、缺失必要的文档字符串、存在无效或错误的注释。这些问题影响代码可读性和可维护性，需要全量修正。

## What Changes
- 对项目全部22个Python文件进行注释格式规范化，采用Google风格文档字符串
- 对项目全部9个Shell脚本文件进行注释格式规范化
- 修正所有与代码逻辑不符的注释内容
- 补全缺失的类、方法、函数注释
- 清理无效、过时、冗余的注释
- **BREAKING**: 无破坏性变更，仅修改注释内容

## Impact
- Affected specs: 代码文档化规范
- Affected code: 
  - Python文件: 22个（models/、eval/、drawer/目录及根目录）
  - Shell文件: 9个（根目录）

## ADDED Requirements

### Requirement: Python类注释规范
所有Python类必须使用Google风格文档字符串(docstring)，必须包含：
- 描述段：清晰说明类的核心功能、适用场景与设计用途
- Attributes段：按「属性名: 类型. 属性含义」格式，完整列出所有公开成员属性

#### Scenario: 类注释完整
- **WHEN** 定义一个Python类
- **THEN** 该类必须包含符合Google风格的docstring，包含描述段和Attributes段

### Requirement: Python方法/函数注释规范
所有def定义的方法、函数必须使用Google风格文档字符串，必须包含：
- 描述段：清晰说明函数/方法的核心功能、执行逻辑与适用场景
- Args段：按「参数名: 类型. 参数含义、取值范围与约束」格式，完整列出所有入参
- Returns段：按「返回值类型. 返回值含义与取值场景」格式，完整描述返回内容
- Raises段（可选）：存在异常抛出时说明异常类型与触发场景

#### Scenario: 函数注释完整
- **WHEN** 定义一个Python函数或方法
- **THEN** 该函数必须包含符合Google风格的docstring，包含描述段、Args段、Returns段

### Requirement: Python行内注释规范
- 代码行上方的单行注释、行尾的内联注释，必须与对应代码逻辑完全匹配
- 行尾注释与代码之间保留2个空格
- 错误的逻辑注释、过时的历史注释、无意义的冗余注释必须修正或删除

#### Scenario: 行内注释规范
- **WHEN** 添加行内注释
- **THEN** 注释内容必须与代码逻辑一致，格式符合Python编码规范

### Requirement: Shell脚本头部注释规范
所有.sh文件必须在shebang行之后补充脚本级注释，包含：
- 脚本核心功能
- 入参说明
- 依赖环境
- 使用场景

#### Scenario: Shell脚本头部注释
- **WHEN** 编写Shell脚本
- **THEN** 脚本头部必须包含功能、入参、依赖、使用场景说明

### Requirement: Shell函数注释规范
Shell脚本中所有自定义函数必须在函数上方补充注释，包含：
- 函数功能
- 入参说明
- 返回值/输出含义

#### Scenario: Shell函数注释
- **WHEN** 定义Shell函数
- **THEN** 函数上方必须有注释说明功能、入参、返回值

### Requirement: Shell行内注释规范
- 脚本中关键逻辑行、分支判断、循环体、复杂命令必须补充清晰的注释说明
- 单行注释以#开头，与注释内容之间保留1个空格
- 无内容的空注释行必须清理

#### Scenario: Shell行内注释
- **WHEN** 添加Shell行内注释
- **THEN** 注释格式统一，内容与命令执行逻辑一致

## MODIFIED Requirements
无修改的需求

## REMOVED Requirements
无移除的需求
