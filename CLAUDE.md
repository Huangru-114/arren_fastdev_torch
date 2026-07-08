# 项目宪法：联邦学习后门攻防文献复现库

> 放置位置：仓库根目录，文件名必须是 `CLAUDE.md`（Claude Code 每次会话开始时自动读取并常驻）。
> 子目录（`strategies/`、`experiments/`、`pfl-lib/`、`papers/`）建议各放一个更简短的 `CLAUDE.md`，
> 只在 Claude Code 实际访问该目录时才按需加载，避免根级上下文过长。

## 0. 项目目标与协作边界

- **目标**：系统性复现联邦学习后门攻击/防御文献，最终产出一套可自由组合（任意攻击 × 任意防御 ×
  任意 FL/pFL 方法 × 任意数据划分）的策略库，同时保留对每篇论文快速判断"是否如文献所述有效"的能力。
- **协作模式**：Claude Code 在本地/沙盒环境工作，与 HPC 之间**只通过 git 同步**，物理上：
  - 无法访问 HPC 文件系统
  - 无法提交/查看 slurm 作业状态
  - 无法得知实验是否正在运行
- 因此 Claude Code 判断"某实验是否完成"的**唯一合法依据**是仓库中是否存在对应的
  `results/<config_hash>/metrics.json`。没有该文件 = 尚未运行，**不得**推测为失败或编造结果。

## 1. 目录结构与依赖方向（四层金字塔 + 只读参考层）

```
papers/            # 只读参考语料，永远不被任何 .py 引用，仅供阅读/提炼
  attacks/atk1/{code, atk1.md}
  defenses/def1/{code, def1.md}
pfl-lib/           # Layer 1 基础设施，只读，只能通过门面调用
  pfl_lib/interface.py
strategies/        # Layer 2 唯一允许写业务逻辑的地方 —— 真正的交付物
  attacks/atk1.py
  defenses/def1.py
  base.py          # AttackHook / DefenseHook 基类
  registry.py       # 装饰器注册机制
experiments/       # Layer 3 编排与验证，不含任何算法逻辑
  configs/
    fidelity/       # 精确复刻论文设置，用于验证有效性
    matrix/         # 攻防组合矩阵，用于广泛探索
  run_single.py
  QUEUE.md          # 跨会话状态追踪表（见第 8 节）
slurm/             # Layer 4 纯调度壳，只生成不提交
  templates/single_run.sbatch.j2
  pending/          # 生成好但尚未被人工提交的 sbatch 文件
  generate_jobs.py
results/           # 精简结果提交入库，重文件本地保留
```

**依赖方向（单向）**：
`papers`（只读参考，禁止 import） + `pfl-lib`（只读，门面） → `strategies`（唯一写代码层） →
`experiments`（编排，组合 strategies） → `slurm`（围绕 experiments 的壳）。禁止反向依赖。

## 2. 修改权限矩阵

| 目录 | Claude Code 权限 |
|---|---|
| `pfl-lib/**` | 未经用户明确许可**不得修改**。需要新功能只能在 `strategies/` 里继承/封装。 |
| `papers/**` | 只读，不可修改，不可被任何 `.py` `import`。 |
| `strategies/**`, `experiments/**`, `slurm/**` | 主要工作区域，可自主创建/修改。 |
| `slurm` 作业提交（`sbatch` 命令本身） | 不存在这个动作——Claude Code 物理上碰不到 HPC，无需额外权限规则。 |

## 3. 导入与代码风格

- 项目内**一律使用绝对导入**（基于包根，如 `from pfl_lib.interface import ...`），不使用相对导入。
- 用 `ruff`/`isort` 强制导入分组与排序，不手工规定"导入必须在前 N 行"之类难以稳定验证的规则。
- `pfl-lib` 通过 `pfl_lib/interface.py` 门面统一暴露模型、数据集、聚合器、数据划分方法；
  **严禁**在 `strategies/` 中直接引用 `pfl-lib` 深层路径。

## 4. 统一入口签名与可组合 Hook 机制

- 每个攻击/防御必须实现 `AttackHook` / `DefenseHook`（定义于 `strategies/base.py`）。
- **攻击 hook 和防御 hook 可以在同一次训练中同时注册**：客户端阶段先执行 attack hook，
  服务端聚合阶段执行 defense hook，执行顺序在框架层固定，不由具体策略代码决定。
- 统一函数签名：`def run(config: ExperimentConfig) -> RunResult`，使 `experiments/run_single.py`
  能无差别调度任意 `(fl_method, partition, attack, defense)` 组合，无需为每个组合手写胶水代码。
- 每个 `strategies/` 文件顶部 docstring 必须注明来源：对应 `papers/` 下的路径 + 论文标题。

## 5. 配置驱动与 schema 校验

- 不允许在 `.py` 中硬编码具体数值（如 `lr=0.01`），一律从 `experiments/configs/*.yaml` 读取。
- 用 dataclass/pydantic 为每类实验定义 config schema，加载时做字段校验，
  避免 yaml 里手滑打错字段名导致运行时静默出错。
- `fidelity/` 下的配置需精确复刻论文原始实验设置（数据集、划分方式、模型、FL 算法）；
  `matrix/` 下的配置用于广泛的攻防组合探索。
- **数据集根目录是 HPC 上的外部资源（如 `/nobackup/proj/disk/.../data`），不属于本仓库，
  仓库内任何 `.py`/`.yaml` 都不得硬编码这个绝对路径。** 统一做法：config schema 中定义一个
  `data_root` 字段，真实路径只在 HPC 侧通过环境变量或命令行参数注入
  （例如 `--data-root $DATA_ROOT`），仓库里的 `fidelity/`、`matrix/` yaml 示例中该字段留空
  或写作占位符（如 `${DATA_ROOT}`），由 `experiments/run_single.py` 在启动时从环境变量解析。

## 6. 两阶段工作流：提炼 → 入库

**阶段一（保真度验证）**：读 `papers/xxx/xxx.md` + 原始代码 → 在 `strategies/` 里实现对应 Hook →
写一份 `experiments/configs/fidelity/xxx_repro.yaml` → 通过 Tier A/B 测试判断是否复现论文效果。

**阶段二（纳入统一库）**：保真度确认后，该策略自动可被 `matrix/` 配置调用，
与任意其他已入库的攻击/防御自由组合，无需额外开发工作。

## 7. 两层测试

- **Tier A · 接线冒烟测试**（Claude Code 自测，不依赖 HPC）：用极小合成数据（如 2 个 client、
  5 个样本、1 轮通信）在 CPU 上跑几秒钟，只验证代码不报错、张量形状正确、hook 确实被调用、
  config 能被正确解析。**每次新增/修改 strategy 后必须当场自测通过才算完成。**
  Tier A 使用的合成数据必须在代码内生成（如随机张量/内置小数据集），**绝不能引用
  `data_root`/HPC 数据目录路径**——Claude Code 沙盒物理上访问不到该路径，若测试中出现
  相关报错，应识别为"路径在此环境不存在"，而不是代码逻辑错误。
- **Tier B · 保真度/性能验证**（必须 HPC，用户手动提交）：判断"论文方法是否真的有效"的
  唯一依据，结果通过 git 回填到 `results/`。
- Claude Code **不得**基于 Tier A 的结果对最终效果做任何结论性断言（如"这个改动应该能提升 ASR"）。

## 8. 结果与状态追踪（git 同步协议）

- `results/<config_hash>/`：提交 `metrics.json`、`summary.md`、`log_tail.txt`（几 KB 量级）；
  `checkpoints/`、完整训练日志等大文件 **gitignore**，仅在 `summary.md` 中记录其在 HPC 上的绝对路径。
- `experiments/QUEUE.md`：唯一跨会话状态表。

  ```markdown
  | config | status | hash | 提交时间 | 备注 |
  |---|---|---|---|---|
  | atk1_repro.yaml | ⏳ pending | a3f9... | - | Tier A 已通过 |
  | atk2_repro.yaml | ✅ done | b71c... | 2026-07-05 | ASR=0.83, 论文报0.85 |
  ```

  Claude Code 每次新增/修改策略后在此登记 `pending` 项；用户手动跑完并 commit 结果后更新为 `done`。
- Claude Code 判断实验状态的唯一依据是仓库中是否存在对应 hash 的 `metrics.json`；
  缺失时应明确说明"等待用户在 HPC 上运行并回填"，**不得**推测成功或失败。
- 用户回填为低频手动操作，无需处理并发写入场景，但生成新 pending 项时应避免覆盖已有 hash 目录。

## 9. slurm 层职责边界

- `slurm/generate_jobs.py` 只读取 `QUEUE.md` 中 `pending` 项，渲染出 `.sh` 脚本（普通 shell
  脚本，用 `sbatch xxx.sh` 提交，**不要用 `.sbatch` 作为文件扩展名**）放入 `slurm/pending/`，
  commit 入库，**不执行**任何提交命令。
- 脚本结构固定，通过 apptainer 容器承载完整 GPU/torch 环境：

  ```bash
  #!/bin/bash
  #SBATCH -n 1
  #SBATCH -c 4
  #SBATCH --gpus 1
  #SBATCH -t {{ time_limit }}
  #SBATCH -A {{ account }}
  #SBATCH -p gpu

  module load GPU/buildenv-nvhpc/25.9-cu13.0
  apptainer exec --nv {{ container_image }} python3 experiments/run_single.py --config {{ config_path }}
  ```

  `{{ container_image }}`、`{{ config_path }}`、`{{ time_limit }}`、`{{ account }}` 由
  `generate_jobs.py` 从 `QUEUE.md` 的 pending 项渲染填入。容器内统一调用
  `experiments/run_single.py`，**不直接调用**各篇论文自己的 `main.py`——后者只作为
  `strategies/` 提炼逻辑时的参考，不再是实际运行入口。
- **容器镜像（如 `torch_fl.sif`）是外部固定依赖，不属于本仓库管理范围**：不需要维护、
  不需要构建定义、Claude Code 不应尝试修改或重建它，只需要在生成脚本时引用其
  已知路径即可。
- Claude Code 沙盒自身的 Tier A 测试环境（见第 7 节）与 HPC 容器环境相互独立，
  不需要保持一致，也不应尝试在沙盒里复现完整 GPU/CUDA 环境。

## 11. 日志规范

- `experiments/run_single.py` 训练循环中，**至少每 10 个 round 输出一次**当前状态到日志，
  内容必须包含：当前 round 数、主任务指标（如 accuracy/loss）、攻击相关指标（如 ASR，
  若适用）、防御相关指标（若适用）、以及**累计训练耗时**（如 `elapsed_time` 或
  `time_per_round`）。
- 格式建议统一为结构化的一行（如 JSON Lines 或固定顺序的 `key=value` 形式），
  便于后续脚本化解析 `log_tail.txt`，而不是自然语言描述句。例如：

  ```
  round=10 acc=0.71 loss=0.83 asr=0.42 elapsed_sec=812.3
  ```

- 这个频率与格式对所有 strategy 统一生效，不由具体攻击/防御的代码自行决定，
  应在 `experiments/run_single.py`（编排层）里统一实现，而不是分散在每个 hook 里各写一遍。

## 10. 何时自主推进，何时停下确认

- 涉及 `pfl-lib/**` 内部修改：**必须停下，等待用户明确许可**。
- 涉及新增/修改 `strategies/`、`experiments/`、`slurm/pending/` 内容：可自主推进，
  完成后在 `QUEUE.md` 登记并完成 Tier A 自测。
- 涉及 Tier B 结果是否存在、实验是否已完成：只依据仓库内 `results/` 文件判断，
  不得猜测，缺失时如实说明"等待用户在 HPC 上运行并回填"。