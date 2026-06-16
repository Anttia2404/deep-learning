from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


def build_scheduler(optimizer: Optimizer, name: str = "reduce_on_plateau", **kwargs):
    key = name.lower()
    if key == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 5),
        )
    if key == "step_lr":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.1),
        )
    raise ValueError(f"Unsupported scheduler: {name}")
