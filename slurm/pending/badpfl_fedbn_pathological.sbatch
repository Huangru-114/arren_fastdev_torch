#!/bin/bash
#SBATCH --job-name=badpfl_fedbn_pathological
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/badpfl_fedbn_pathological-%j.out

set -euo pipefail

# Repo root: submit from the repo root, or `export REPO_ROOT=/path/to/repo` first.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# ---- environment preparation (edit for your cluster) ----
module load Anaconda3
conda activate arren_fastdev_torch

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/pfl-lib:${PYTHONPATH:-}"
cd "$REPO_ROOT"

# ---- the only algorithmic entry point ----
python experiments/run_single.py --config "experiments/configs/matrix/badpfl_fedbn_pathological.yaml"