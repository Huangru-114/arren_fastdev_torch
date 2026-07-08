#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 24:00:00
#SBATCH -A NAISS-2025-EXAMPLE
#SBATCH -p gpu
module load GPU/buildenv-nvhpc/25.9-cu13.0
apptainer exec --nv /nobackup/containers/torch_fl.sif python3 experiments/run_single.py --config experiments/configs/fidelity/badpfl_fedbn_repro.yaml