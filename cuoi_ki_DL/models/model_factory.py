from __future__ import annotations

from typing import Any

import torch.nn as nn

from .meso_net import MesoInception4, MesoNet
from .xception import build_xception


def create_model(name: str, **kwargs: Any) -> nn.Module:
    key = name.lower()
    if key == "xception":
        return build_xception(
            pretrained=kwargs.get("pretrained", True),
            num_classes=kwargs.get("num_classes", 2),
            dropout=kwargs.get("dropout", 0.5),
        )
    if key == "mesonet":
        return MesoNet(num_classes=kwargs.get("num_classes", 1), input_size=kwargs.get("input_size", 256))
    if key == "mesoinception4":
        return MesoInception4(num_classes=kwargs.get("num_classes", 1), input_size=kwargs.get("input_size", 256))
    raise ValueError(f"Unsupported model: {name}")
