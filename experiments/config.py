"""Config schema loading and validation (Layer 3).

Reads a YAML file into a validated :class:`ExperimentConfig`. Validation is
deliberately strict: unknown keys (typos) are rejected rather than silently
ignored, enum-like fields are checked against their allowed values, and the
partition strength that matters for the chosen scheme is required.

A stable ``config_hash`` is derived from the normalized config (excluding
runtime-only fields like ``device``) and used to name the ``results/`` directory.
"""

import dataclasses
import hashlib
import json

import yaml

from strategies.base import ExperimentConfig

_ALLOWED_PARTITIONS = {"dirichlet", "pathological"}
_ALLOWED_AGG = {"avg", "average", "median"}
# fields excluded from the identity hash (do not change experiment semantics)
_HASH_EXCLUDE = {"device"}


def _field_names():
    return {f.name for f in dataclasses.fields(ExperimentConfig)}


def load_config(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return from_dict(raw)


def from_dict(raw):
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    allowed = _field_names()
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown config field(s): {sorted(unknown)}; allowed: {sorted(allowed)}")
    if "name" not in raw:
        raise ValueError("config must define 'name'")

    config = ExperimentConfig(**raw)
    _validate(config)
    return config


def _validate(c: ExperimentConfig):
    if c.partition not in _ALLOWED_PARTITIONS:
        raise ValueError(f"partition must be one of {sorted(_ALLOWED_PARTITIONS)}, got {c.partition!r}")
    if c.agg_rule not in _ALLOWED_AGG:
        raise ValueError(f"agg_rule must be one of {sorted(_ALLOWED_AGG)}, got {c.agg_rule!r}")
    if c.partition == "dirichlet" and c.dir_alpha <= 0:
        raise ValueError("dir_alpha must be > 0 for dirichlet partition")
    if c.partition == "pathological" and not (1 <= c.class_per_client <= c.num_classes):
        raise ValueError("class_per_client must be in [1, num_classes] for pathological partition")
    if c.bad_client_num < 0 or c.bad_client_num > c.client_num:
        raise ValueError("bad_client_num must be in [0, client_num]")
    if c.select_client_num_per_round < 1 or c.select_client_num_per_round > c.client_num:
        raise ValueError("select_client_num_per_round must be in [1, client_num]")
    if c.client_local_step < 1:
        raise ValueError("client_local_step must be >= 1")


def config_hash(c: ExperimentConfig) -> str:
    d = {k: v for k, v in dataclasses.asdict(c).items() if k not in _HASH_EXCLUDE}
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
