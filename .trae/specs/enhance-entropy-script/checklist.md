# Checklist

## 配置验证
- [x] ROPE 和 ADAPTER 标志已添加并可正常切换
- [x] ROPE_METHODS 格式已更新为完整参数字符串格式
- [x] ADAPTER_DIR 配置已添加
- [x] ADAPTER_PATHS 数组已添加并包含示例配置
- [x] METHODS 数组构建逻辑正确，能根据标志组合配置

## 功能验证
- [x] entropy.py 支持 --adapter-path 参数（或已添加支持）
- [x] ROPE=true, ADAPTER=false 时仅评估 RoPE 方法
- [x] ROPE=false, ADAPTER=true 时仅评估 Adapter
- [x] ROPE=true, ADAPTER=true 时评估所有配置
- [x] 参数正确传递给 entropy.py 脚本

## 代码质量
- [x] 脚本结构与 eval.sh 保持一致
- [x] 注释清晰，说明各配置项的作用
- [x] 错误处理完善
- [x] 输出信息清晰，便于调试
