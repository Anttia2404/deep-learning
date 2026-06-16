from __future__ import annotations

import torch.nn as nn


def build_loss(loss_name: str = "cross_entropy", label_smoothing: float = 0.0):
    key = loss_name.lower()
    if key == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if key == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unsupported loss: {loss_name}")
