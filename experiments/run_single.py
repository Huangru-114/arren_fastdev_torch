"""Unified single-run entry point (Layer 3).

Dispatches any ``(fl_method, partition, attack, defense)`` combination described by
a YAML config with no combination-specific glue:

    python experiments/run_single.py --config experiments/configs/fidelity/badpfl_fedbn_repro.yaml

Contains no attack/defense/pFL algorithm logic -- it only wires infrastructure
(``pfl_lib.interface``) to the strategy hooks (``strategies``) and drives the
engine, then writes a compact ``results/<hash>/`` record.
"""

import argparse
import json
import os
import random
import sys
from dataclasses import asdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Two source roots: repo root (strategies/, experiments/) and pfl-lib/ (pfl_lib pkg).
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "pfl-lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import strategies.all  # noqa: F401,E402  (registers built-in strategies)
from experiments.config import config_hash, load_config  # noqa: E402
from pfl_lib.interface import build_dataset, build_model, partition_data  # noqa: E402
from strategies.base import ExperimentConfig, FLEngine, RunResult  # noqa: E402
from strategies.registry import get_attack, get_defense, get_pfl  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_root(config: ExperimentConfig) -> str:
    """Expand ``config.data_root`` against the environment (constitution s.5).

    ``${DATA_ROOT}`` -> the DATA_ROOT env var. Real (non-synthetic) datasets
    require a resolved path; the ``synthetic`` dataset needs none (Tier A).
    """
    resolved = os.path.expandvars(config.data_root or "")
    if config.dataset == "synthetic":
        return resolved  # unused by the synthetic provider
    if not resolved or "${" in resolved:
        raise ValueError(
            f"data_root is unresolved ({config.data_root!r}); the dataset root is an "
            "external HPC path -- inject it via the DATA_ROOT env var or --data-root "
            "(constitution s.5). It is never hardcoded in the repo."
        )
    return resolved


def run(config: ExperimentConfig) -> RunResult:
    set_seed(config.seed)

    bundle = build_dataset(
        config.dataset,
        root=resolve_data_root(config),
        synthetic_spec=config.synthetic,
    )

    train_idx = partition_data(
        bundle.train_targets, config.client_num, bundle.num_classes, config.partition,
        alpha=config.dir_alpha, class_per_client=config.class_per_client, balance=config.balance,
    )
    test_idx = partition_data(
        bundle.test_targets, config.client_num, bundle.num_classes, config.partition,
        alpha=config.dir_alpha, class_per_client=config.class_per_client, balance=config.balance,
    )

    def model_factory():
        return build_model(config.model, num_classes=bundle.num_classes,
                           input_channel=bundle.input_channel)

    pfl = get_pfl(config.fl_method)(**config.fl_params)
    attack = get_attack(config.attack)(**config.attack_params) if config.attack else None
    defense = get_defense(config.defense)(**config.defense_params) if config.defense else None

    engine = FLEngine(config, bundle, train_idx, test_idx, model_factory, pfl, attack, defense)
    engine.run()
    accs, asrs = engine.evaluate()

    acc_t = torch.tensor(accs) if accs else torch.zeros(1)
    asr_t = torch.tensor(asrs) if asrs else torch.zeros(1)
    return RunResult(
        name=config.name,
        config_hash=config_hash(config),
        acc_mean=round(acc_t.mean().item(), 4),
        acc_std=round(acc_t.std().item(), 4),
        asr_mean=round(asr_t.mean().item(), 4) if asrs else 0.0,
        asr_std=round(asr_t.std().item(), 4) if asrs else 0.0,
        per_client_acc=[round(a, 4) for a in accs],
        per_client_asr=[round(a, 4) for a in asrs],
        extra={"num_eval_clients": len(accs)},
    )


def write_results(config: ExperimentConfig, result: RunResult):
    out_dir = os.path.join(_REPO_ROOT, "results", result.config_hash)
    os.makedirs(out_dir, exist_ok=True)
    metrics = {
        "name": result.name,
        "config_hash": result.config_hash,
        "acc_mean": result.acc_mean,
        "acc_std": result.acc_std,
        "asr_mean": result.asr_mean,
        "asr_std": result.asr_std,
        "config": asdict(config),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(
            f"# {result.name}\n\n"
            f"- hash: `{result.config_hash}`\n"
            f"- fl_method: {config.fl_method} | partition: {config.partition} "
            f"(alpha={config.dir_alpha}, class_per_client={config.class_per_client})\n"
            f"- attack: {config.attack} | defense: {config.defense}\n"
            f"- **Acc**: {result.acc_mean:.2f} ± {result.acc_std:.2f}\n"
            f"- **ASR**: {result.asr_mean:.2f} ± {result.asr_std:.2f}\n"
            f"- eval clients: {result.extra.get('num_eval_clients')}\n\n"
            f"Heavy artifacts (checkpoints/full logs) are kept on HPC; record their "
            f"absolute paths here when backfilling.\n"
        )
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None,
                        help="dataset root (overrides DATA_ROOT / config.data_root); external HPC path")
    parser.add_argument("--no-write", action="store_true", help="run but do not write results/ (smoke tests)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root is not None:
        config.data_root = args.data_root
    result = run(config)
    print(
        f"[{result.name}] hash={result.config_hash} "
        f"Acc={result.acc_mean:.2f}±{result.acc_std:.2f} "
        f"ASR={result.asr_mean:.2f}±{result.asr_std:.2f} "
        f"(eval clients={result.extra.get('num_eval_clients')})"
    )
    if not args.no_write:
        out_dir = write_results(config, result)
        print(f"results written to {out_dir}")


if __name__ == "__main__":
    main()
