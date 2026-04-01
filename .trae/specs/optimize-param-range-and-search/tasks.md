# Tasks

## Task 1: 扩大参数搜索范围
- [x] Task 1.1: 更新 `search_attn_scale_params.py` 中的参数默认值：`attn_scale_coef_min=0.01`、`attn_scale_coef_max=0.3`
- [x] Task 1.2: 更新 `search_params.sh` 中的配置：`COEF_MIN=0.01`、`COEF_MAX=0.3`

## Task 2: 增加采样密度
- [x] Task 2.1: 更新网格搜索采样点数量：从 50 增加到 100
- [x] Task 2.2: 更新随机搜索样本数量：从 30 增加到 50
- [x] Task 2.3: 更新贝叶斯优化迭代次数：从 50 增加到 100
- [x] Task 2.4: 更新自适应搜索阶段数：从 4-5 增加到 5-6

## Task 3: 实现 BOHB 搜索策略
- [x] Task 3.1: 添加 BOHB 搜索方法选项
- [x] Task 3.2: 实现初始随机采样阶段（20-30 个配置）
- [x] Task 3.3: 实现低保真度评估（减少样本数）
- [x] Task 3.4: 实现早停机制（淘汰表现差的配置）
- [x] Task 3.5: 实现贝叶斯优化推荐新配置

## Task 4: 实现多阶段搜索
- [x] Task 4.1: 实现初始探索阶段（随机采样）
- [x] Task 4.2: 实现精细搜索阶段（贝叶斯优化）
- [x] Task 4.3: 实现保真度切换机制
- [x] Task 4.4: 添加阶段转换日志

## Task 5: 优化评估流程
- [x] Task 5.1: 实现多保真度评估（低保真度：25 样本，高保真度：50 样本）
- [x] Task 5.2: 添加评估缓存机制
- [x] Task 5.3: 优化内存管理
- [x] Task 5.4: 添加并行评估支持

## Task 6: 更新配置文件和文档
- [x] Task 6.1: 更新 `search_params.sh`，添加新的搜索策略选项
- [x] Task 6.2: 更新命令行参数帮助信息
- [x] Task 6.3: 添加搜索策略对比文档
- [x] Task 6.4: 更新示例配置

## Task 7: 测试和验证
- [x] Task 7.1: 测试网格搜索（100 个采样点）
- [x] Task 7.2: 测试 BOHB 搜索策略
- [x] Task 7.3: 验证参数范围覆盖已知最优值（0.0707、0.1）
- [x] Task 7.4: 对比不同搜索策略的效果

# Task Dependencies
- [Task 2] 依赖 [Task 1]（增加采样密度需要先扩大参数范围）
- [Task 3] 依赖 [Task 1] 和 [Task 2]（BOHB 需要完整的参数配置）
- [Task 4] 依赖 [Task 3]（多阶段搜索需要 BOHB 基础）
- [Task 5] 依赖 [Task 4]（优化评估需要多阶段搜索）
- [Task 6] 依赖 [Task 1] 到 [Task 5]（配置文件需要所有功能完成）
- [Task 7] 依赖 [Task 1] 到 [Task 6]（测试需要所有功能完成）
