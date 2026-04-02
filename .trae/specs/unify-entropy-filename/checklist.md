# Checklist

## 功能验证
- [x] generate_save_filename 函数支持 adapter_path 参数
- [x] Adapter-only 模式生成正确的文件名（包含 adapter 标识符）
- [x] RoPE-only 模式生成正确的文件名（与现有格式一致）
- [x] Adapter+RoPE 组合模式生成正确的文件名（包含两者信息）
- [x] 文件名格式与 perplexity.py、performance.py 保持一致

## 代码质量
- [x] 函数文档清晰，包含所有使用场景的示例
- [x] 代码逻辑简洁，易于理解
- [x] 错误处理完善（处理空路径、无效路径等情况）
- [x] 向后兼容（不破坏现有功能）

## 测试验证
- [x] 使用不同 adapter 路径测试文件名生成
- [x] 使用不同 RoPE 配置测试文件名生成
- [x] 使用组合配置测试文件名生成
