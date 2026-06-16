from __future__ import annotations

import torch.nn as nn
import timm


class XceptionClassifier(nn.Module):
    """Xception backbone with dropout head for binary deepfake detection."""

    def __init__(self, pretrained: bool = True, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.backbone = timm.create_model("xception", pretrained=pretrained, num_classes=0, global_pool="avg")
        num_features = getattr(self.backbone, "num_features", 2048)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def build_xception(pretrained: bool = True, num_classes: int = 2, dropout: float = 0.5) -> nn.Module:
    return XceptionClassifier(pretrained=pretrained, num_classes=num_classes, dropout=dropout)
