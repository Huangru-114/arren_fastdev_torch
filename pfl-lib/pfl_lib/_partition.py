"""Client data partitioning (Layer 1 infrastructure).

Two non-IID schemes are provided, both ported from PFLlib
(``PFLlib/dataset/utils/dataset_utils.py`` ``separate_data``) and re-expressed as
pure index partitioners that return ``{client_id: np.ndarray[int]}``:

- ``dirichlet``     -> label proportions drawn from ``Dir(alpha)`` per class.
                       Strength knob: ``alpha`` (smaller == more skewed).
- ``pathological``  -> each client sees only ``class_per_client`` classes,
                       samples of a class are split among the clients that own it.
                       Strength knob: ``class_per_client`` (smaller == more skewed).

Both are seeded through numpy's global RNG so callers control reproducibility via
``numpy.random.seed`` upstream.
"""

import numpy as np

_DEFAULT_MIN_REQUIRE_SIZE = 10


def dirichlet_partition(targets, num_clients, num_classes, alpha, min_require_size=_DEFAULT_MIN_REQUIRE_SIZE):
    """Dirichlet label-skew partition (PFLlib ``partition='dir'``)."""
    targets = np.asarray(targets)
    n = targets.shape[0]
    min_size = 0
    idx_batch = [[] for _ in range(num_clients)]

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # cap clients that already hold more than their even share
            proportions = np.array(
                [p * (len(idx_j) < n / num_clients) for p, idx_j in zip(proportions, idx_batch)]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min(len(idx_j) for idx_j in idx_batch)

    return {cid: np.asarray(idx_batch[cid], dtype=np.int64) for cid in range(num_clients)}


def pathological_partition(targets, num_clients, num_classes, class_per_client, balance=True):
    """Pathological partition (PFLlib ``partition='pat'``)."""
    targets = np.asarray(targets)
    idxs = np.arange(len(targets))
    idx_for_each_class = [idxs[targets == i] for i in range(num_classes)]

    dataidx_map = {}
    class_num_per_client = [class_per_client for _ in range(num_clients)]
    for i in range(num_classes):
        selected_clients = [c for c in range(num_clients) if class_num_per_client[c] > 0]
        if len(selected_clients) == 0:
            break
        selected_clients = selected_clients[: int(np.ceil((num_clients / num_classes) * class_per_client))]

        num_all_samples = len(idx_for_each_class[i])
        num_selected_clients = len(selected_clients)
        num_per = num_all_samples / num_selected_clients
        if balance:
            num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
        else:
            low = max(int(num_per / 10), 1)
            num_samples = np.random.randint(low, max(low + 1, int(num_per)), num_selected_clients - 1).tolist()
        num_samples.append(num_all_samples - sum(num_samples))

        idx = 0
        for client, num_sample in zip(selected_clients, num_samples):
            block = idx_for_each_class[i][idx: idx + num_sample]
            if client not in dataidx_map:
                dataidx_map[client] = block
            else:
                dataidx_map[client] = np.append(dataidx_map[client], block, axis=0)
            idx += num_sample
            class_num_per_client[client] -= 1

    return {cid: np.asarray(dataidx_map.get(cid, np.array([], dtype=np.int64)), dtype=np.int64) for cid in range(num_clients)}


def partition_data(targets, num_clients, num_classes, scheme, alpha=0.5, class_per_client=2, balance=True):
    """Dispatch to a named partition scheme. ``scheme`` in {"dirichlet", "pathological"}."""
    if scheme == "dirichlet":
        return dirichlet_partition(targets, num_clients, num_classes, alpha=alpha)
    if scheme == "pathological":
        return pathological_partition(
            targets, num_clients, num_classes, class_per_client=class_per_client, balance=balance
        )
    raise ValueError(f"unknown partition scheme {scheme!r}; choose 'dirichlet' or 'pathological'")
