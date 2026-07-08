# experiments/ —— Layer 3 编排与验证（不含算法逻辑）

- `run_single.py`：统一入口 `run(config) -> RunResult`。按 config 把基础设施
  (`pfl_lib.interface`) 接到策略 hook (`strategies`)，驱动 `FLEngine`，评估 Acc/ASR，
  写 `results/<hash>/{metrics.json,summary.md}`。任意 `(fl_method, partition, attack, defense)`
  组合无需专门胶水代码。
- `config.py`：YAML → 校验后的 `ExperimentConfig`。未知字段（打错字）直接报错；
  枚举字段校验；`config_hash` 决定 `results/` 目录名（排除 `device`、`data_root` 等运行期字段）。
- **数据集根目录（`data_root`，宪法 s.5）**：外部 HPC 路径，仓库内绝不硬编码。yaml 里写占位符
  `${DATA_ROOT}`，`run_single.py` 启动时按 `--data-root` > `DATA_ROOT` 环境变量的优先级解析；
  真实数据集未解析则报错，`synthetic`（Tier A）不需要它。因排除出 hash，注入路径不影响 `results/` 目录名。
- `configs/fidelity/`：精确复刻论文设置（Bad-PFL×FedBN/FedRep，CIFAR-10，dir 0.5）。
- `configs/matrix/`：攻防×划分组合探索（含 pathological 与更强 dirichlet）。
- `smoke_test.py`：Tier A 接线冒烟测试（合成数据、CPU、数秒）；改动策略后必须当场通过。
- `QUEUE.md`：唯一跨会话状态表。判定"完成"只看 `results/<hash>/metrics.json` 是否存在。

数据分布轴：`partition: dirichlet|pathological`，强度旋钮分别是 `dir_alpha` 与 `class_per_client`。
