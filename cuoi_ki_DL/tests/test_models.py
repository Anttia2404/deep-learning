from __future__ import annotations

import torch

from models.model_factory import create_model


def test_xception_forward_shape():
    model = create_model("xception", pretrained=False, num_classes=2, dropout=0.5)
    x = torch.randn(2, 3, 299, 299)
    y = model(x)
    assert y.shape == (2, 2)


def test_mesonet_forward_shape():
    model = create_model("mesonet", num_classes=2, input_size=256)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    assert y.shape == (2, 2)


def test_mesoinception_forward_shape():
    model = create_model("mesoinception4", num_classes=2, input_size=256)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    assert y.shape == (2, 2)
