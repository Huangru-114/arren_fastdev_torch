"""Bad-PFL backdoor attack (AttackHook).

Source: papers/attacks/atk1/Bad-PFL --
  "Bad-PFL: Exploring Backdoor Attacks against Personalized Federated Learning",
  Fan et al., ICLR 2025 (poster).

Core idea reproduced here: the trigger ``T(x) = delta + xi`` combines a
target-feature perturbation ``delta = epsilon * G_w(x)`` produced by a learned
encoder-decoder generator, and a disruptive noise ``xi`` crafted as a one-step
PGD/FGSM perturbation that pushes the model to rely on ``delta`` rather than on
the sample's own features. On each malicious client the generator is trained for
``gen_steps`` steps (Adam) to map triggered inputs to the target label, then the
local model is trained with a fraction ``poison_rate`` of each batch replaced by
triggered, target-labelled samples.

All numeric defaults come from the paper (s.3.7): epsilon = sigma = 4/255,
poison_rate = 0.2, generator lr = 0.01, gen_steps = 30. They are overridable via
``attack_params`` in the experiment config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from strategies.base import AttackHook
from strategies.registry import register_attack


class _Autoencoder(nn.Module):
    """Encoder-decoder trigger generator (Bad-PFL Table 5).

    Four stride-2 conv layers down, four transpose-conv layers up, ``Tanh`` output
    so ``G(x)`` lies in ``[-1, 1]`` and ``epsilon * G(x)`` respects the L-inf budget.
    Input spatial size must be divisible by 16.
    """

    def __init__(self, channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(16, channels, 4, stride=2, padding=1), nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _pgd_noise(model, images, labels, epsilon, alpha, num_iter):
    """Disruptive noise xi: adversarial perturbation within an L-inf ball of the
    clean image, crafted to *maximize* the clean-task loss (one FGSM step by
    default). Returns ``x + xi`` clamped to valid pixel range."""
    adv = images.clone().detach() + torch.zeros_like(images).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv, 0.0, 1.0)
    for _ in range(num_iter):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), labels)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv + alpha * grad.sign()
        eta = torch.clamp(adv - images, -epsilon, epsilon)
        adv = torch.clamp(images + eta, 0.0, 1.0).detach()
    return adv.detach()


@register_attack("bad_pfl")
class BadPFLAttack(AttackHook):
    def __init__(self, **params):
        super().__init__(**params)
        self.target_label = int(params.get("target_label", 0))
        self.poison_rate = float(params.get("poison_rate", 0.2))
        self.epsilon = float(params.get("epsilon", 4.0 / 255.0))
        self.sigma = float(params.get("sigma", 4.0 / 255.0))
        self.gen_lr = float(params.get("gen_lr", 1e-2))
        self.gen_steps = int(params.get("gen_steps", 30))
        self.pgd_iter = int(params.get("pgd_iter", 1))
        self.generator = None
        self.gen_optimizer = None

    # -- generator: delta = epsilon * G(x) --
    def _delta(self, x):
        return self.epsilon * self.generator(x)

    def setup(self, engine):
        device = engine.device
        channels = engine.bundle.input_channel
        self.generator = _Autoencoder(channels).to(device)
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr)

        for client in engine.clients:
            if client.is_malicious:
                client.register("before_local_training", self._train_generator)
                client.poison_func = self._make_poison_func(client)

    def _train_generator(self, client):
        """Optimize G so that triggered inputs are classified as the target label
        (Bad-PFL eq. 7), against the client's current model (kept in eval mode)."""
        self.generator.train()
        client.local_model.eval()
        for _ in range(self.gen_steps):
            clean_data, clean_label = client.fetch_clean_data()
            self.gen_optimizer.zero_grad()
            adv = _pgd_noise(client.local_model, clean_data, clean_label,
                             self.sigma, self.sigma, self.pgd_iter)
            triggered = adv + self._delta(clean_data)
            pred = client.local_model(triggered)
            target = torch.full((clean_label.size(0),), self.target_label,
                                dtype=torch.long, device=clean_label.device)
            loss = F.cross_entropy(pred, target)
            loss.backward()
            self.gen_optimizer.step()

    def _make_poison_func(self, client):
        def poison_func(data, label):
            return self._apply_trigger(client.local_model, data, label, self.poison_rate)
        return poison_func

    @torch.no_grad()
    def _apply_trigger(self, model, data, label, poison_rate):
        mask = torch.rand(label.size(0), device=label.device) <= poison_rate
        if mask.sum().item() == 0:
            return data, label
        with torch.enable_grad():
            adv = _pgd_noise(model, data, label, self.sigma, self.sigma, self.pgd_iter)
        triggered = adv + self._delta(data)
        m = mask.view(-1, 1, 1, 1).float()
        poisoned_data = m * triggered + (1.0 - m) * data
        target = torch.full_like(label, self.target_label)
        poisoned_label = torch.where(mask, target, label)
        return poisoned_data, poisoned_label

    def eval_transform(self):
        # ASR transform: trigger every sample (poison_rate = 1) against the given
        # personalized model, relabel to the target class.
        def transform(inputs, labels, model):
            return self._apply_trigger(model, inputs, labels, poison_rate=1.0)
        return transform
