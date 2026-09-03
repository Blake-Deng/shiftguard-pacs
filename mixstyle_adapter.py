"""Faithful MixStyle layer adapted from Zhou et al. (ICLR 2021).

Source: KaiyangZhou/mixstyle-release, reid/models/mixstyle.py.
"""
from __future__ import annotations

import random
import torch
from torch import nn


class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6, mix: str = "random"):
        super().__init__()
        self.p, self.alpha, self.eps, self.mix = p, alpha, eps, mix
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p:
            return x
        batch = x.size(0)
        mu = x.mean(dim=(2, 3), keepdim=True)
        sig = (x.var(dim=(2, 3), keepdim=True) + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        normalized = (x - mu) / sig
        lam = self.beta.sample((batch, 1, 1, 1)).to(x.device)
        if self.mix == "random":
            perm = torch.randperm(batch, device=x.device)
        elif self.mix == "crossdomain":
            perm = torch.arange(batch - 1, -1, -1, device=x.device)
        else:
            raise ValueError(f"unsupported MixStyle pairing: {self.mix}")
        mu2, sig2 = mu[perm], sig[perm]
        return normalized * (sig * lam + sig2 * (1.0 - lam)) + (mu * lam + mu2 * (1.0 - lam))


def inject_resnet50(model: nn.Module, p: float = 0.5, alpha: float = 0.1) -> nn.Module:
    """Insert MixStyle after ResNet-50 layer1 and layer2."""
    if not hasattr(model, "layer1") or not hasattr(model, "layer2"):
        raise TypeError("MixStyle adapter requires a torchvision ResNet")
    model.layer1 = nn.Sequential(model.layer1, MixStyle(p=p, alpha=alpha))
    model.layer2 = nn.Sequential(model.layer2, MixStyle(p=p, alpha=alpha))
    return model
