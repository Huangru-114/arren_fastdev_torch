"""Dataset providers (Layer 1 infrastructure).

Real datasets are loaded through torchvision with a plain ``ToTensor`` transform
(pixels in ``[0, 1]`` as Bad-PFL assumes). A ``synthetic`` provider backs the
Tier-A smoke test: it needs no download and runs in seconds on CPU.

Every provider returns a ``DatasetBundle`` exposing ``train``/``test`` datasets,
integer ``targets`` for both splits (needed by the partitioners), ``num_classes``
and ``input_channel``.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetBundle:
    train: Dataset
    test: Dataset
    train_targets: list
    test_targets: list
    num_classes: int
    input_channel: int
    image_size: int


class _SyntheticImages(Dataset):
    """Deterministic random images with class-correlated bias, for smoke tests."""

    def __init__(self, num_samples, num_classes, channels, size, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=g)
        base = torch.rand(num_samples, channels, size, size, generator=g)
        # nudge each image toward a per-class mean so accuracy is learnable
        bias = (self.labels.float() / max(num_classes - 1, 1)).view(-1, 1, 1, 1)
        self.data = (0.5 * base + 0.5 * bias).clamp(0, 1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], int(self.labels[idx])


def _torchvision_bundle(name, root, download):
    import torchvision
    import torchvision.datasets as tvds

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if name == "cifar10":
        train = tvds.CIFAR10(root, train=True, download=download, transform=transform)
        test = tvds.CIFAR10(root, train=False, download=download, transform=transform)
        return DatasetBundle(train, test, list(train.targets), list(test.targets), 10, 3, 32)
    if name == "cifar100":
        train = tvds.CIFAR100(root, train=True, download=download, transform=transform)
        test = tvds.CIFAR100(root, train=False, download=download, transform=transform)
        return DatasetBundle(train, test, list(train.targets), list(test.targets), 100, 3, 32)
    if name == "svhn":
        train = tvds.SVHN(root, split="train", download=download, transform=transform)
        test = tvds.SVHN(root, split="test", download=download, transform=transform)
        return DatasetBundle(train, test, list(train.labels), list(test.labels), 10, 3, 32)
    raise ValueError(f"unknown torchvision dataset {name!r}")


def build_dataset(name, root="./data", download=True, synthetic_spec=None):
    """Build a :class:`DatasetBundle`.

    ``name='synthetic'`` uses ``synthetic_spec`` (a dict) for a tiny in-memory
    dataset; any other name is delegated to torchvision.
    """
    if name == "synthetic":
        spec = synthetic_spec or {}
        num_classes = spec.get("num_classes", 4)
        channels = spec.get("input_channel", 3)
        size = spec.get("image_size", 32)
        n_train = spec.get("num_train", 64)
        n_test = spec.get("num_test", 32)
        train = _SyntheticImages(n_train, num_classes, channels, size, seed=1)
        test = _SyntheticImages(n_test, num_classes, channels, size, seed=2)
        return DatasetBundle(
            train, test,
            [int(v) for v in train.labels], [int(v) for v in test.labels],
            num_classes, channels, size,
        )
    return _torchvision_bundle(name, root, download)
