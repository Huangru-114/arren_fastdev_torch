"""FedBN (Li et al., ICLR 2021): keep batch-norm layers strictly local.

Reference for the reproduction target:
  papers/attacks/atk1/Bad-PFL -- "Bad-PFL: Exploring Backdoor Attacks against
  Personalized Federated Learning" (ICLR 2025), which evaluates FedBN as one of
  the partial-model-sharing pFL baselines.

Partial sharing rule: parameters whose key contains ``bn`` or ``shortcut.1``
(the downsample BatchNorm) are private -- never uploaded, never overwritten by
the server -- so each client keeps its own normalization statistics.
"""

from strategies.base import PFLMethod
from strategies.registry import register_pfl


@register_pfl("fedbn")
class FedBN(PFLMethod):
    def is_private(self, key: str) -> bool:
        return ("bn" in key) or ("shortcut.1" in key)
