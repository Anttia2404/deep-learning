from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.ff_dataset import FaceForensicsDataset
from datasets.transforms import build_transforms
from evaluation.metrics import compute_classification_metrics
from models.model_factory import create_model
from training.trainer import TrainingConfig, resolve_device


def load_checkpoint_model(checkpoint_path: Path, model_name: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_data = checkpoint.get("config", {})
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=config_data.get("num_classes", 2),
        dropout=config_data.get("dropout", 0.5),
        input_size=config_data.get("input_size", 299 if model_name == "xception" else 256),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_frame_level(checkpoint_path: Path, data_root: Path, model_name: str, manipulation: str, compression: str, batch_size: int = 32, device_name: str | None = None):
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint_model(checkpoint_path, model_name, device)
    transforms = build_transforms(model_name)
    dataset = FaceForensicsDataset(
        data_root=data_root,
        split="test",
        manipulation=manipulation,
        compression=compression,
        transform=transforms["test"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            logits = model(images)
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
            probs.extend(batch_probs)
            labels.extend(batch_labels.numpy().tolist())

    metrics = compute_classification_metrics(labels, probs)
    return {
        "checkpoint_val_acc": checkpoint.get("val_acc"),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FaceForensics++ checkpoint on test split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", default="xception", choices=["xception", "mesonet", "mesoinception4"])
    parser.add_argument("--manipulation", default="Deepfakes")
    parser.add_argument("--compression", default="c23")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    results = evaluate_frame_level(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        model_name=args.model,
        manipulation=args.manipulation,
        compression=args.compression,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
