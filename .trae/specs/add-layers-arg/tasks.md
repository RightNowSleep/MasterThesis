# Tasks

- [x] Task 1: 修改 `_auto_select_layers` 函数，增加 `preferred_layers` 参数
  - [x] 增加 `preferred_layers: List[int] | None = None` 参数
  - [x] 对 preferred_layers 去重、过滤越界索引
  - [x] 若 preferred_layers 长度 >= n，截取前 n 个返回
  - [x] 若 preferred_layers 长度 < n，用原 std 降序规则补充不重复的层至 n 个
  - [x] 若 preferred_layers 为 None 或空，保持原行为不变

- [x] Task 2: 修改 `plot_entropy_vs_position` 函数，增加 `preferred_layers` 参数
  - [x] 增加 `preferred_layers: List[int] | None = None` 参数
  - [x] 当 `selected_layers` 为 None 时，将 `preferred_layers` 传递给 `_auto_select_layers`

- [x] Task 3: 修改 `plot_all` 函数，增加 `preferred_layers` 参数
  - [x] 增加 `preferred_layers: List[int] | None = None` 参数
  - [x] 将 `preferred_layers` 传递给 `_auto_select_layers` 和 `plot_entropy_vs_position`

- [x] Task 4: 修改 `_build_parser`，增加 `--layers` 命令行参数
  - [x] 添加 `--layers` 参数，`type=int`，`nargs="*"`，`default=None`
  - [x] 添加 help 文本说明用法

- [x] Task 5: 修改 `main` 函数，解析并传递 `--layers`
  - [x] 从 `args.layers` 获取用户指定的层列表
  - [x] 单图模式：将 `preferred_layers` 传递给 `_auto_select_layers` 和 `plot_entropy_vs_position`
  - [x] 全图模式：将 `preferred_layers` 传递给 `plot_all`

- [x] Task 6: 修改 `entropy.sh`，增加 `PLOT_LAYERS` 配置
  - [x] 添加 `PLOT_LAYERS` 变量（默认为空）
  - [x] 在 Part 2 的 python 调用中，当 `PLOT_LAYERS` 非空时追加 `--layers $PLOT_LAYERS`

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 5 depends on Task 3 and Task 4
- Task 6 is independent of Tasks 1–5 (can be parallelized)
