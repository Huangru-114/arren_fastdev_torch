"""FedRep (Collins et al., ICML 2021): shared representation, private head.

Reference for the reproduction target:
  papers/attacks/atk1/Bad-PFL -- "Bad-PFL: Exploring Backdoor Attacks against
  Personalized Federated Learning" (ICLR 2025), which evaluates FedRep as one of
  the partial-model-sharing pFL baselines.

Partial sharing rule: the classification head (parameters whose key contains
``linear``) is private. Local training is two-phase per the FedRep algorithm:
first update the head with the body frozen, then update the body (representation)
with the head frozen. The relative number of head steps is configurable via
``fl_params.head_steps`` (default: half of the local steps, at least one each).
"""

from strategies.base import Client, PFLMethod
from strategies.registry import register_pfl


@register_pfl("fedrep")
class FedRep(PFLMethod):
    def is_private(self, key: str) -> bool:
        return "linear" in key

    def _split_params(self, model):
        head, body = [], []
        for name, param in model.named_parameters():
            (head if "linear" in name else body).append(param)
        return head, body

    @staticmethod
    def _set_requires_grad(params, flag):
        for p in params:
            p.requires_grad_(flag)

    def local_train(self, client: Client, local_steps: int) -> None:
        head_steps = int(self.params.get("head_steps", max(1, local_steps // 2)))
        head_steps = max(1, min(head_steps, local_steps - 1)) if local_steps > 1 else local_steps
        body_steps = local_steps - head_steps

        head, body = self._split_params(client.local_model)

        # Phase 1: train the private head, representation frozen.
        self._set_requires_grad(body, False)
        self._set_requires_grad(head, True)
        for _ in range(head_steps):
            client.train_step(client.local_model, client.optimizer)

        # Phase 2: train the shared representation, head frozen.
        self._set_requires_grad(head, False)
        self._set_requires_grad(body, True)
        for _ in range(body_steps):
            client.train_step(client.local_model, client.optimizer)

        # restore for safety (e.g. attack generator training uses full model)
        self._set_requires_grad(head, True)
        self._set_requires_grad(body, True)
