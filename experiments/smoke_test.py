"""Tier-A wiring smoke test (constitution s.7). CPU, seconds, no HPC.

Runs every (fl_method x partition) combination on a tiny synthetic dataset and
asserts only that the plumbing is correct -- no errors, sane shapes/metrics,
attack hooks actually fire, and partial-model-sharing trims the right parameters.
It deliberately makes NO claim about attack effectiveness (that is Tier B).

    python experiments/smoke_test.py
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "pfl-lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import strategies.all  # noqa: F401,E402
from experiments.run_single import run  # noqa: E402
from strategies.base import ExperimentConfig  # noqa: E402
from strategies.registry import get_attack, get_pfl  # noqa: E402


def _base_config(fl_method, partition, **overrides):
    kw = dict(
        name=f"smoke_{fl_method}_{partition}",
        dataset="synthetic",
        num_classes=4,
        input_channel=3,
        partition=partition,
        dir_alpha=0.5,
        class_per_client=2,
        fl_method=fl_method,
        model="resnet10",
        client_num=4,
        bad_client_num=2,
        select_client_num_per_round=2,
        total_round=2,
        client_local_step=3,
        client_batch=4,
        learning_rate=0.05,
        attack="bad_pfl",
        attack_params={"gen_steps": 2, "target_label": 1},
        seed=0,
        device="cpu",
        synthetic={"num_classes": 4, "num_train": 96, "num_test": 48, "image_size": 32},
    )
    if fl_method == "fedrep":
        kw["fl_params"] = {"head_steps": 2}
    kw.update(overrides)
    return ExperimentConfig(**kw)


def test_partial_sharing_keys():
    """FedBN hides BN params; FedRep hides the head -- verified against a real model."""
    from pfl_lib.interface import build_model

    model = build_model("resnet10", num_classes=4)
    keys = list(model.state_dict().keys())

    fedbn = get_pfl("fedbn")()
    hidden_bn = [k for k in keys if fedbn.is_private(k)]
    assert hidden_bn, "FedBN should hide some BN parameters"
    assert all(("bn" in k) or ("shortcut.1" in k) for k in hidden_bn)
    assert "linear.weight" not in hidden_bn, "FedBN must still share the head"

    fedrep = get_pfl("fedrep")()
    hidden_head = [k for k in keys if fedrep.is_private(k)]
    assert set(hidden_head) == {"linear.weight", "linear.bias"}, hidden_head
    # trim_private must actually drop them from an uploaded dict
    trimmed = fedrep.trim_private(model.state_dict())
    assert "linear.weight" not in trimmed and "bn1.weight" in trimmed
    print("  [ok] partial-sharing key selection (FedBN BN / FedRep head)")


def test_attack_wiring():
    """Malicious clients (and only those) get a poison_func + generator trainer."""
    from pfl_lib.interface import build_dataset, build_model, partition_data
    from strategies.base import FLEngine

    config = _base_config("fedbn", "dirichlet")
    bundle = build_dataset("synthetic", synthetic_spec=config.synthetic)
    train_idx = partition_data(bundle.train_targets, config.client_num, bundle.num_classes,
                               "dirichlet", alpha=0.5)
    test_idx = partition_data(bundle.test_targets, config.client_num, bundle.num_classes,
                              "dirichlet", alpha=0.5)
    attack = get_attack("bad_pfl")(**config.attack_params)
    pfl = get_pfl("fedbn")()
    engine = FLEngine(config, bundle, train_idx, test_idx,
                      lambda: build_model("resnet10", num_classes=4), pfl, attack, None)

    mal = [c for c in engine.clients if c.is_malicious]
    ben = [c for c in engine.clients if not c.is_malicious]
    assert len(mal) == config.bad_client_num
    assert all(c.poison_func is not None for c in mal), "malicious clients need a poison_func"
    assert all(c.poison_func is None for c in ben), "benign clients must stay clean"
    assert all("before_local_training" in c._stage_hooks for c in mal), "generator trainer not wired"
    assert attack.generator is not None, "generator not built in setup()"
    print("  [ok] attack wiring (poison_func + generator trainer on malicious clients only)")


def test_end_to_end():
    for fl_method in ("fedbn", "fedrep"):
        for partition in ("dirichlet", "pathological"):
            config = _base_config(fl_method, partition)
            result = run(config)
            assert result.extra["num_eval_clients"] >= 1, "no clients had test data"
            assert 0.0 <= result.acc_mean <= 100.0, result.acc_mean
            assert 0.0 <= result.asr_mean <= 100.0, result.asr_mean
            assert result.per_client_asr, "ASR not computed -- eval transform missing"
            assert result.config_hash and len(result.config_hash) == 12
            print(f"  [ok] {fl_method:6s} x {partition:12s} "
                  f"Acc={result.acc_mean:5.2f} ASR={result.asr_mean:5.2f} "
                  f"(clients={result.extra['num_eval_clients']})")


if __name__ == "__main__":
    print("Tier-A smoke test:")
    test_partial_sharing_keys()
    test_attack_wiring()
    test_end_to_end()
    print("ALL SMOKE TESTS PASSED")
