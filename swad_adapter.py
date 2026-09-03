"""Epoch-granularity source-validation SWAD adapter.

The valley rule and defaults follow the official SWAD LossValley
implementation (khanrc/swad/domainbed/swad.py). The current trainer exposes
one source-validation observation per epoch, so checkpoints are treated as
epoch segments; no target data are loaded during selection.
"""
from __future__ import annotations

from copy import deepcopy

import torch


class LossValleyEpoch:
    def __init__(self, n_converge: int = 3, n_tolerance: int = 6, tolerance_ratio: float = 0.3):
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio
        self.history = []
        self.converged_at = None
        self.start_epoch = None
        self.threshold = None

    def update(self, epoch: int, state: dict[str, torch.Tensor], val_loss: float) -> None:
        item = (epoch, deepcopy(state), float(val_loss))
        self.history.append(item)
        if self.converged_at is None and len(self.history) >= self.n_converge:
            window = self.history[-self.n_converge:]
            losses = [x[2] for x in window]
            min_idx = min(range(len(losses)), key=losses.__getitem__)
            if min_idx == 0:
                self.converged_at = epoch
                self.start_epoch = window[0][0]
                self.threshold = sum(losses) / len(losses) * (1.0 + self.tolerance_ratio)

    def final(self, fallback: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int | None, int | None]:
        if self.converged_at is None:
            if self.history:
                epoch, state, _ = self.history[-1]
                return state, epoch, epoch
            return fallback, None, None
        selected = []
        after_start = [item for item in self.history if item[0] >= self.start_epoch]
        for index, item in enumerate(after_start):
            selected.append(item)
            if len(selected) >= self.n_tolerance:
                window = selected[-self.n_tolerance:]
                if min(x[2] for x in window) > self.threshold:
                    selected = selected[:-self.n_tolerance]
                    break
        if not selected:
            return fallback, None, None
        keys = selected[0][1].keys()
        averaged = {}
        for key in keys:
            value = selected[0][1][key]
            if torch.is_floating_point(value):
                averaged[key] = sum(item[1][key] for item in selected) / len(selected)
            else:
                averaged[key] = value
        return averaged, selected[0][0], selected[-1][0]
