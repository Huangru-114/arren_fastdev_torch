"""Framework contract for the strategy library (Layer 2).

This module defines, once and for all, *the order in which things happen* during
a federated run, so that individual strategies never have to. It provides:

- :class:`ExperimentConfig` / :class:`RunResult` -- the unified I/O of a run.
- :class:`PFLMethod` / :class:`AttackHook` / :class:`DefenseHook` -- the three
  extension points. A run composes at most one of each.
- :class:`Client` / :class:`Server` / :class:`FLEngine` -- the algorithm-agnostic
  harness that drives them.

Execution order fixed by the framework (constitution s.4): on each selected
client the attack hook runs during local training (client stage); on the server
the defense hook runs during aggregation (server stage). pFL partial-sharing is
applied by trimming private parameters on both upload and distribute.

Only ``pfl_lib.interface`` may be imported from the infrastructure layer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, Subset


# --------------------------------------------------------------------------- #
# Config / result schemas
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    """Validated description of a single federated run.

    Field validation lives in ``experiments/config.py`` (schema loader); this
    dataclass is the in-memory shape the engine consumes.
    """

    name: str
    # data
    dataset: str = "cifar10"
    # dataset root is an external HPC resource (constitution s.5): never a hardcoded
    # absolute path. Keep the ``${DATA_ROOT}`` placeholder here; the real path is
    # injected at launch from the DATA_ROOT env var or ``--data-root``. Unused by
    # the ``synthetic`` dataset (Tier A generates data in-memory).
    data_root: str = "${DATA_ROOT}"
    num_classes: int = 10
    input_channel: int = 3
    partition: str = "dirichlet"          # 'dirichlet' | 'pathological'
    dir_alpha: float = 0.5                 # dirichlet strength
    class_per_client: int = 2              # pathological strength
    balance: bool = True
    # federation
    fl_method: str = "fedbn"               # registered pFL method
    model: str = "resnet10"
    client_num: int = 100
    bad_client_num: int = 10
    select_client_num_per_round: int = 10
    total_round: int = 300
    client_local_step: int = 15
    client_batch: int = 32
    learning_rate: float = 0.1
    agg_rule: str = "avg"
    # attack / defense
    attack: Optional[str] = "bad_pfl"      # registered attack or None
    defense: Optional[str] = None          # registered defense or None
    attack_params: dict = field(default_factory=dict)
    defense_params: dict = field(default_factory=dict)
    fl_params: dict = field(default_factory=dict)
    # runtime
    seed: int = 2024
    device: str = "cpu"
    # synthetic-only knobs (Tier-A smoke test)
    synthetic: dict = field(default_factory=dict)


@dataclass
class RunResult:
    name: str
    config_hash: str
    acc_mean: float
    acc_std: float
    asr_mean: float
    asr_std: float
    per_client_acc: list = field(default_factory=list)
    per_client_asr: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Extension points
# --------------------------------------------------------------------------- #
class PFLMethod:
    """Personalized-FL method: decides which parameters stay private and, if
    needed, how a client trains locally."""

    name: str = "base"

    def __init__(self, **params):
        self.params = params

    def is_private(self, key: str) -> bool:
        """Return True for parameters that are NOT shared with the server."""
        return False

    def trim_private(self, state_dict: dict) -> dict:
        return {k: v for k, v in state_dict.items() if not self.is_private(k)}

    def local_train(self, client: "Client", local_steps: int) -> None:
        """Default local training; override for multi-phase schemes (e.g. FedRep)."""
        for _ in range(local_steps):
            client.train_step(client.local_model, client.optimizer)


class AttackHook:
    """Backdoor attack. Attached to malicious clients only."""

    name: str = "base"

    def __init__(self, **params):
        self.params = params

    def setup(self, engine: "FLEngine") -> None:
        """Called once after clients/server exist; wire malicious clients here."""

    def eval_transform(self) -> Optional[Callable]:
        """Return an ``(inputs, labels) -> (inputs, labels)`` transform used to
        build triggered inputs for ASR evaluation, or None if not applicable."""
        return None


class DefenseHook:
    """Robust-aggregation / post-aggregation defense (server stage)."""

    name: str = "base"

    def __init__(self, **params):
        self.params = params

    def on_aggregate(self, aggregated: dict, client_updates: list, server: "Server") -> dict:
        return aggregated


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
class Client:
    def __init__(self, cid, model, dataloader, optimizer_factory, device, is_malicious=False):
        self.cid = cid
        self.local_model = model
        self.local_model.device = device
        self.device = device
        self.dataloader = dataloader
        self._iter = iter(dataloader) if dataloader is not None else None
        self.optimizer = optimizer_factory(self.local_model.parameters())
        self.optimizer_factory = optimizer_factory
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.is_malicious = is_malicious
        self.poison_func: Optional[Callable] = None       # set by AttackHook
        self._stage_hooks: dict = {}

    # -- hook registry (per-client) --
    def register(self, stage: str, func: Callable):
        self._stage_hooks.setdefault(stage, []).append(func)

    def _emit(self, stage: str):
        for func in self._stage_hooks.get(stage, []):
            func(self)

    # -- data --
    def _raw_batch(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            batch = next(self._iter)
        return batch[0].to(self.device), batch[1].to(self.device)

    def fetch_clean_data(self):
        """Raw batch with no attack transform (used e.g. to train the trigger generator)."""
        return self._raw_batch()

    def fetch_data(self):
        data, label = self._raw_batch()
        if self.poison_func is not None:
            data, label = self.poison_func(data, label)
        return data, label

    # -- training --
    def receive(self, global_state_dict):
        self.local_model.load_state_dict(global_state_dict, strict=False)
        self._emit("before_local_training")

    def train_step(self, model, optimizer):
        model.train()
        optimizer.zero_grad()
        data, label = self.fetch_data()
        loss = self.loss_func(model(data), label)
        loss.backward()
        self._emit("before_update")
        optimizer.step()
        return loss

    def upload(self, pfl: PFLMethod):
        return pfl.trim_private(self.local_model.state_dict())


class Server:
    def __init__(self, global_model, aggregator, pfl: PFLMethod, defense: Optional[DefenseHook], device):
        self.global_model = global_model
        self.global_model.device = device
        self.aggregator = aggregator
        self.pfl = pfl
        self.defense = defense
        self.device = device

    @torch.no_grad()
    def distribute(self):
        return self.pfl.trim_private(self.global_model.state_dict())

    @torch.no_grad()
    def aggregate(self, client_updates):
        aggregated = self.aggregator(client_updates)
        if self.defense is not None:
            aggregated = self.defense.on_aggregate(aggregated, client_updates, self)
        self.global_model.load_state_dict(aggregated, strict=False)


class FLEngine:
    """Algorithm-agnostic federated training loop."""

    def __init__(self, config: ExperimentConfig, dataset_bundle, client_indices,
                 test_indices, model_factory, pfl: PFLMethod,
                 attack: Optional[AttackHook], defense: Optional[DefenseHook]):
        self.config = config
        self.bundle = dataset_bundle
        self.device = torch.device(config.device)
        self.pfl = pfl
        self.attack = attack
        self.defense = defense
        self.model_factory = model_factory

        from pfl_lib.interface import get_aggregator

        opt_factory = lambda params: torch.optim.SGD(params, lr=config.learning_rate)

        # malicious cids: the last ``bad_client_num`` client ids
        malicious = set(range(config.client_num - config.bad_client_num, config.client_num))
        self.clients = []
        for cid in range(config.client_num):
            train_loader = DataLoader(
                Subset(self.bundle.train, list(client_indices[cid])),
                batch_size=config.client_batch, shuffle=True, drop_last=False,
            )
            self.clients.append(
                Client(cid, model_factory().to(self.device), train_loader, opt_factory,
                       self.device, is_malicious=cid in malicious)
            )
        # per-client personalized test loaders (for Acc / ASR)
        self.client_test_loaders = [
            DataLoader(Subset(self.bundle.test, list(test_indices[cid])),
                       batch_size=64, shuffle=False)
            for cid in range(config.client_num)
        ]

        self.server = Server(model_factory().to(self.device),
                             get_aggregator(config.agg_rule), pfl, defense, self.device)

        if attack is not None:
            attack.setup(self)

    def select(self):
        k = self.config.select_client_num_per_round
        return torch.randperm(self.config.client_num)[:k].tolist()

    def run(self):
        for _ in range(self.config.total_round):
            global_state = self.server.distribute()
            selected = self.select()
            updates = []
            for cid in selected:
                client = self.clients[cid]
                client.receive(copy.deepcopy(global_state))
                self.pfl.local_train(client, self.config.client_local_step)
                updates.append(client.upload(self.pfl))
            self.server.aggregate(updates)

    # -- evaluation --
    @torch.no_grad()
    def _accuracy(self, model, loader, transform=None):
        model.eval()
        correct = total = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if transform is not None:
                inputs, labels = transform(inputs, labels)
            pred = model(inputs).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        return 100.0 * correct / max(total, 1)

    def evaluate(self):
        # The attack's eval transform builds triggered inputs against a *specific*
        # model, so it is bound to each client's own personalized model here.
        base_transform = self.attack.eval_transform() if self.attack is not None else None
        accs, asrs = [], []
        for cid, client in enumerate(self.clients):
            loader = self.client_test_loaders[cid]
            if len(loader.dataset) == 0:
                continue
            accs.append(self._accuracy(client.local_model, loader))
            if base_transform is not None:
                model = client.local_model
                transform = lambda inp, lab, _m=model: base_transform(inp, lab, _m)
                asrs.append(self._accuracy(client.local_model, loader, transform=transform))
        return accs, asrs
