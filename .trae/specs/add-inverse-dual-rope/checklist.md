# Checklist

- [x] pe_llama.py 中 LlamaInverseDualRoPEEmbedding 类实现完整且正确
  - [x] __init__ 方法正确初始化所有参数
  - [x] i_star 计算逻辑正确
  - [x] inv_freq 正确分割为高频和低频两部分
  - [x] _set_cos_sin_cache 中高频区域使用 t // L_0
  - [x] _set_cos_sin_cache 中低频区域使用 t % L_0
  - [x] 分母 L_0 与 scaling_factor 无关
  - [x] forward 方法支持动态和静态模式

- [x] pe_llama.py 中 LlamaInverseDualRoPEScaledEmbedding 类实现完整且正确
  - [x] 继承自 LlamaInverseDualRoPEEmbedding
  - [x] _compute_attn_scale 使用正确的 log 公式
  - [x] 温度缩放正确应用到 cos/sin 缓存

- [x] configuration_llama.py 更新完成
  - [x] valid_types 包含 "inverse-dual-rope"
  - [x] valid_types 包含 "inverse-dual-rope-scaled"
  - [x] 文档字符串已更新

- [x] model_loader.py 更新完成
  - [x] _ROPE_TYPES_WITH_DYNAMIC_FLAG 包含两个新类型
  - [x] _ALL_ROPE_TYPES 列表包含新类型
  - [x] help 文本包含新类型说明

- [x] modeling_llama.py 集成完成
  - [x] "inverse-dual-rope" 路由到正确类
  - [x] "inverse-dual-rope-scaled" 路由到正确类
  - [x] 错误消息包含新类型

- [x] continued_pretrain.py 渐进式训练功能实现
  - [x] --progressive-length 参数可用
  - [x] 长度序列生成算法正确 (2的幂次增长)
  - [x] 训练循环支持多阶段长度切换
  - [x] 阶段间过渡平滑（可选）
  - [x] 日志输出显示当前长度阶段

- [x] continued_pretrain.sh 配置更新
  - [x] PROGRESSIVE_LENGTH 变量定义
  - [x] 参数正确传递给 Python 脚本

- [x] 功能测试通过
  - [x] Inverse-Dual-RoPE 可正常实例化
  - [x] 前向传播输出形状正确
  - [x] Scaled 版本缩放因子计算正确
  - [x] 渐进式训练流程可正常运行
