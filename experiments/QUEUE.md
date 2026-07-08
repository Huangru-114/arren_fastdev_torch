# 实验状态追踪表（唯一跨会话状态源）

> 判定规则（宪法 s.8）：某实验"完成"的唯一依据是仓库中存在对应 hash 的
> `results/<hash>/metrics.json`。缺失 = 尚未运行，不得推测成功或失败。
> Tier A（接线冒烟）由 Claude Code 本地跑 `python experiments/smoke_test.py`；
> Tier B（保真/性能）必须在 HPC 手动提交，结果经 git 回填后把状态改为 ✅ done。

| config | status | hash | 提交时间 | 备注 |
|---|---|---|---|---|
| badpfl_fedbn_repro.yaml | ⏳ pending | f2082981fbca | - | Bad-PFL×FedBN 保真，CIFAR-10 dir(0.5)，论文报 Acc80.72/ASR82.22；Tier A 已通过 |
| badpfl_fedrep_repro.yaml | ⏳ pending | 49ad3f547abb | - | Bad-PFL×FedRep 保真，CIFAR-10 dir(0.5)，论文报 Acc80.29/ASR97.95；Tier A 已通过 |
| badpfl_fedbn_pathological.yaml | ⏳ pending | dd8b1e1429fa | - | 矩阵探索，FedBN×pathological(class_per_client=2)；Tier A 已通过 |
| badpfl_fedrep_dirichlet_strong.yaml | ⏳ pending | 1ab46ac21925 | - | 矩阵探索，FedRep×dir(0.1) 强偏移；Tier A 已通过 |

## 数据分布轴（本次扩充新增）
- `partition: dirichlet`，强度旋钮 `dir_alpha`（越小越偏，默认 0.5）。
- `partition: pathological`，强度旋钮 `class_per_client`（越小越偏，默认 2）。
- 两者均来自 `pfl_lib.interface.partition_data`（移植自 PFLlib 的 `dir`/`pat`）。
