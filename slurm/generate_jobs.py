"""Render sbatch scripts for pending experiments (Layer 4 -- schedule shell only).

Reads ``experiments/QUEUE.md``, finds rows marked pending, locates the matching
YAML under ``experiments/configs/**``, and renders one ``.sbatch`` per pending
config into ``slurm/pending/``. It NEVER submits anything (Claude Code has no HPC
access); a human submits the generated files with ``sbatch`` manually.

    python slurm/generate_jobs.py [--time 24:00:00] [--mem 32G] [--gpus 1]
"""

import argparse
import os
import re

from jinja2 import Template

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_QUEUE = os.path.join(_REPO_ROOT, "experiments", "QUEUE.md")
_CONFIG_ROOT = os.path.join(_REPO_ROOT, "experiments", "configs")
_TEMPLATE = os.path.join(_REPO_ROOT, "slurm", "templates", "single_run.sbatch.j2")
_PENDING = os.path.join(_REPO_ROOT, "slurm", "pending")

_DEFAULT_ENV_SETUP = "module load Anaconda3\nconda activate arren_fastdev_torch"


def parse_pending(queue_path):
    """Return the config filenames of rows whose status contains 'pending'."""
    pending = []
    with open(queue_path, "r") as f:
        for line in f:
            if not line.strip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 2:
                continue
            config_cell, status_cell = cells[0], cells[1]
            if config_cell in ("config", "---") or set(config_cell) <= {"-", ":"}:
                continue
            if "pending" in status_cell.lower():
                pending.append(config_cell)
    return pending


def find_config(filename):
    for dirpath, _dirs, files in os.walk(_CONFIG_ROOT):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None


def render(args):
    with open(_TEMPLATE, "r") as f:
        template = Template(f.read())

    os.makedirs(_PENDING, exist_ok=True)
    pending = parse_pending(_QUEUE)
    if not pending:
        print("no pending configs in QUEUE.md")
        return

    generated = []
    for filename in pending:
        config_path = find_config(filename)
        if config_path is None:
            print(f"  [skip] {filename}: not found under experiments/configs/")
            continue
        job_name = re.sub(r"\.ya?ml$", "", filename)
        config_rel_path = os.path.relpath(config_path, _REPO_ROOT)
        rendered = template.render(
            job_name=job_name,
            gpus=args.gpus,
            mem=args.mem,
            time=args.time,
            log_dir=args.log_dir,
            env_setup=args.env_setup,
            config_rel_path=config_rel_path,
        )
        out_path = os.path.join(_PENDING, f"{job_name}.sbatch")
        with open(out_path, "w") as f:
            f.write(rendered)
        generated.append(out_path)
        print(f"  [ok] {filename} -> {os.path.relpath(out_path, _REPO_ROOT)}")

    print(f"\nGenerated {len(generated)} sbatch file(s) in slurm/pending/ (NOT submitted).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--env-setup", default=_DEFAULT_ENV_SETUP)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
