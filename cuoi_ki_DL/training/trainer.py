from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.ff_dataset import FaceForensicsDataset, FaceForensicsVideoDataset, build_balanced_sampler
from datasets.transforms import build_transforms
from models.model_factory import create_model
from training.losses import build_loss
from training.schedulers import build_scheduler


@dataclass
class TrainingConfig:
    model_name: str = "xception"
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.5
    input_size: int = 299
    data_root: str = "./data/FaceForensics++"
    manipulation: str = "Deepfakes"
    compression: str = "c23"
    batch_size: int = 32
    num_epochs: int = 30
    num_frames_train: int = 270
    num_frames_eval: int = 270
    lr: float = 2e-4
    weight_decay: float = 0.0
    scheduler: str = "reduce_on_plateau"
    loss_name: str = "cross_entropy"
    label_smoothing: float = 0.0
    num_workers: int = 0
    checkpoint_dir: str = "./checkpoints"
    early_stopping_patience: int = 10
    seed: int = 42
    use_video_dataset: bool = False
    detector_backend: str = "mtcnn"
    device: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: Path) -> TrainingConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    flat = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return TrainingConfig(**flat)


def make_datasets(config: TrainingConfig):
    transforms = build_transforms(config.model_name)
    dataset_cls = FaceForensicsVideoDataset if config.use_video_dataset else FaceForensicsDataset
    common_kwargs = dict(
        data_root=config.data_root,
        manipulation=config.manipulation,
        compression=config.compression,
    )
    if config.use_video_dataset:
        train_ds = dataset_cls(
            split="train",
            num_frames=min(config.num_frames_train, 32),
            transform=transforms["train"],
            detector_backend=config.detector_backend,
            detector_device="cuda" if resolve_device(config.device).type == "cuda" else None,
            face_size=config.input_size,
            **common_kwargs,
        )
        val_ds = dataset_cls(
            split="val",
            num_frames=min(config.num_frames_eval, 32),
            transform=transforms["val"],
            detector_backend=config.detector_backend,
            detector_device="cuda" if resolve_device(config.device).type == "cuda" else None,
            face_size=config.input_size,
            **common_kwargs,
        )
    else:
        train_ds = dataset_cls(
            split="train",
            num_frames=config.num_frames_train,
            transform=transforms["train"],
            **common_kwargs,
        )
        val_ds = dataset_cls(
            split="val",
            num_frames=config.num_frames_eval,
            transform=transforms["val"],
            **common_kwargs,
        )
    return train_ds, val_ds


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1), all_probs, all_labels


def save_checkpoint(state: dict[str, Any], checkpoint_dir: Path, filename: str) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    return path


def run_training(config: TrainingConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    train_dataset, val_dataset = make_datasets(config)

    sampler = build_balanced_sampler(train_dataset) if config.manipulation == "all" and not config.use_video_dataset else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(
        config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        dropout=config.dropout,
        input_size=config.input_size,
    ).to(device)

    criterion = build_loss(config.loss_name, label_smoothing=config.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, name=config.scheduler)

    best_val_acc = -1.0
    best_path = None
    stale_epochs = 0
    history = []

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        if config.scheduler.lower() == "reduce_on_plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record))

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "config": asdict(config),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            },
            Path(config.checkpoint_dir),
            filename=f"latest_{config.manipulation}_{config.compression}.pth",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stale_epochs = 0
            best_path = save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "config": asdict(config),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                Path(config.checkpoint_dir),
                filename=f"best_{config.manipulation}_{config.compression}.pth",
            )
        else:
            stale_epochs += 1

        if stale_epochs >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return {
        "best_val_acc": best_val_acc,
        "best_checkpoint": str(best_path) if best_path else None,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FaceForensics++ classifiers")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--manipulation", type=str, default=None)
    parser.add_argument("--compression", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config.data_root = args.data_root
    if args.manipulation is not None:
        config.manipulation = args.manipulation
    if args.compression is not None:
        config.compression = args.compression
    if args.device is not None:
        config.device = args.device

    results = run_training(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
