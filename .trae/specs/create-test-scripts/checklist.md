# Checklist

## test.sh 验证
- [x] test.sh 脚本具有正确的执行权限
- [x] test.sh 支持 perplexity 子命令并正确调用 test.py
- [x] test.sh 支持 passkey 子命令并正确调用 test.py
- [x] test.sh 支持 quality 子命令并正确调用 test.py
- [x] test.sh 支持 performance 子命令并正确调用 test.py
- [x] test.sh 正确传递所有RoPE相关参数（--rope-type、--rope-factor、--rope-dynamic）
- [x] test.sh 正确传递模型相关参数（--model-name、--max-length、--min-length）
- [x] test.sh 包含使用帮助信息

## eval.sh 验证
- [x] eval.sh 脚本具有正确的执行权限
- [x] eval.sh 支持 perplexity 子命令并正确调用 eval/perplexity.py
- [x] eval.sh 支持 passkey 子命令并正确调用 eval/passkey.py
- [x] eval.sh 支持 quality 子命令并正确调用 eval/quality.py
- [x] eval.sh 支持 performance 子命令并正确调用 eval/performance.py
- [x] eval.sh 正确传递所有RoPE相关参数（--rope-type、--rope-factor、--rope-dynamic）
- [x] eval.sh 正确传递模型相关参数（--model-name、--max-length、--min-length）
- [x] eval.sh 包含使用帮助信息

## 通用验证
- [x] 两个脚本都支持所有RoPE类型（none、linear、ntk、part-ntk、yarn、my-rope、dynamic-my-rope）
- [x] 两个脚本的参数命名与Python脚本中的参数一致
