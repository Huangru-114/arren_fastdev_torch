"""Public facade for the ``pfl_lib`` Layer-1 infrastructure.

Everything the upper layers (``strategies``/``experiments``) are allowed to use
from the infrastructure must be imported from *this* module. Deep imports such as
``pfl_lib._models`` are forbidden by the project constitution — go through the
names re-exported here:

- models       : :func:`build_model`
- datasets     : :func:`build_dataset`, :class:`DatasetBundle`
- partitioning : :func:`partition_data` (dirichlet / pathological, adjustable strength)
- aggregation  : :func:`get_aggregator`
"""

from pfl_lib._aggregators import get_aggregator
from pfl_lib._datasets import DatasetBundle, build_dataset
from pfl_lib._models import build_resnet
from pfl_lib._partition import partition_data


def build_model(arch="resnet10", num_classes=10, input_channel=3):
    """Build a backbone by name. Currently ``resnet10``/``resnet18``/``resnet34``."""
    if not arch.startswith("resnet"):
        raise ValueError(f"unknown model arch {arch!r}")
    size = int(arch[len("resnet"):])
    return build_resnet(size=size, input_channel=input_channel, num_classes=num_classes)


__all__ = [
    "build_model",
    "build_dataset",
    "DatasetBundle",
    "partition_data",
    "get_aggregator",
]
