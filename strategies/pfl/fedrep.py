"""FedRep (Collins et al., ICML 2021): shared representation, private head.

Reference for the reproduction target:
  papers/attacks/atk1/Bad-PFL -- "Bad-PFL: Exploring Backdoor Attacks against
  Personalized Federated Learning" (ICLR 2025), which evaluates FedRep as one of
  the partial-model-sharing pFL baselines.

Partial sharing rule: the classification head (parameters whose key contains
``linear``) is private. Local training is two-phase per the FedRep algorithm
(cf. PFLlib ``clientRep.train``): first update the head for a SEPARATE
``head_steps`` budget with the body frozen, then update the representation for a
FULL local epoch (``local_steps``) with the head frozen. Head and body use
separate SGD optimizers. Only the representation is uploaded and aggregated.

``fl_params.head_steps`` sets the head budget (default: half of ``local_steps``).
The representation always gets the full ``local_steps`` -- it is NOT reduced by
the head phase. (An earlier version starved it to ``local_steps - head_steps``,
which under-trained the shared features and collapsed both Acc and ASR.)
"""

import torch

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
        head, body = self._split_params(client.local_model)
        head_steps = max(1, int(self.params.get("head_steps", max(1, local_steps // 2))))

        # Separate optimizers for head and body (PFLlib clientRep: optimizer_per
        # over head params, optimizer over base params). lr is read from the
        # client's optimizer, so it stays config-driven and is never hardcoded.
        lr = client.optimizer.param_groups[0]["lr"]
        head_optimizer = torch.optim.SGD(head, lr=lr)
        body_optimizer = torch.optim.SGD(body, lr=lr)

        # Phase 1 (head): freeze the representation, train the private head for a
        # separate head_steps budget on the full local data.
        self._set_requires_grad(body, False)
        self._set_requires_grad(head, True)
        for _ in range(head_steps):
            client.train_step(client.local_model, head_optimizer)

        # Phase 2 (body): freeze the head, train the shared representation for the
        # FULL local budget (local_steps) -- NOT local_steps - head_steps. The
        # representation must get a full local epoch (PFLlib clientRep:
        # max_local_epochs = self.local_epochs).
        self._set_requires_grad(head, False)
        self._set_requires_grad(body, True)
        for _ in range(local_steps):
            client.train_step(client.local_model, body_optimizer)

        # restore for anything that reuses the full model (e.g. attack generator).
        self._set_requires_grad(head, True)
        self._set_requires_grad(body, True)
