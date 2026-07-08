# slurm/ —— Layer 4 纯调度壳（只生成，不提交）

- `generate_jobs.py`：读 `experiments/QUEUE.md` 里的 `pending` 行，找到对应 yaml，按固定模板
  渲染 `.sbatch` 到 `slurm/pending/`，**从不执行 sbatch**（Claude Code 物理上碰不到 HPC）。
- `templates/single_run.sbatch.j2`：结构固定（宪法 s.9），用 apptainer 容器承载 GPU/torch 环境：
  `apptainer exec --nv {{ container_image }} python3 experiments/run_single.py --config {{ config_path }}`。
  占位符：`time_limit`/`account`/`container_image` 为集群参数，由 `generate_jobs.py` 的 CLI 参数注入
  （`--time`/`--account`/`--container-image`，默认是 `${SLURM_ACCOUNT}`/`${CONTAINER_IMAGE}` 占位符）；
  `config_path` 为仓库相对路径，从 pending 项解析。
- 容器镜像（如 `torch_fl.sif`）是**外部固定依赖**：不构建、不修改、不重建，只引用其已知路径。
- 沙盒 Tier A 环境与 HPC 容器环境相互独立，不必一致；不要在沙盒里复现完整 GPU/CUDA 环境。
- 容器内统一调用 `experiments/run_single.py`，**不直接调用** `papers/` 下各论文的 `main.py`。
