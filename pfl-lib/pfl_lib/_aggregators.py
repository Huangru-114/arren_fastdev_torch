"""Server-side aggregation rules (Layer 1 infrastructure).

Each aggregator maps a list of client ``state_dict``s to a single aggregated
``state_dict``. Only parameters present in *every* client's dict are aggregated;
this lets partial-model-sharing pFL methods (FedBN/FedRep) upload trimmed dicts
without the aggregator choking on missing keys.
"""

import torch


def _common_keys(state_dicts):
    keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        keys &= set(sd.keys())
    # preserve the ordering of the first dict for determinism
    return [k for k in state_dicts[0].keys() if k in keys]


def agg_average(state_dicts):
    keys = _common_keys(state_dicts)
    out = {}
    for key in keys:
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        out[key] = stacked.mean(dim=0).to(state_dicts[0][key].dtype)
    return out


def agg_median(state_dicts):
    keys = _common_keys(state_dicts)
    out = {}
    for key in keys:
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        out[key] = stacked.median(dim=0).values.to(state_dicts[0][key].dtype)
    return out


_AGGREGATORS = {
    "avg": agg_average,
    "average": agg_average,
    "median": agg_median,
}


def get_aggregator(name):
    if name not in _AGGREGATORS:
        raise ValueError(f"unknown aggregator {name!r}; choose from {sorted(_AGGREGATORS)}")
    return _AGGREGATORS[name]
