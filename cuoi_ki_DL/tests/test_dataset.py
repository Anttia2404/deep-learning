from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from datasets.ff_dataset import FaceForensicsDataset, build_balanced_sampler
from datasets.transforms import build_transforms


def _write_image(path: Path) -> None:
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _build_fixture(root: Path) -> None:
    (root / "splits").mkdir(parents=True, exist_ok=True)
    (root / "splits" / "train.json").write_text(json.dumps(["000", "001"]), encoding="utf-8")
    (root / "splits" / "val.json").write_text(json.dumps(["000"]), encoding="utf-8")
    (root / "splits" / "test.json").write_text(json.dumps(["001"]), encoding="utf-8")

    real_dir = root / "original_sequences" / "youtube" / "c23" / "faces"
    fake_dir = root / "manipulated_sequences" / "Deepfakes" / "c23" / "faces"
    for video_id in ["000", "001"]:
        target = real_dir / video_id
        target.mkdir(parents=True, exist_ok=True)
        _write_image(target / "00000.png")
        _write_image(target / "00001.png")

    pair_dir = fake_dir / "000_001"
    pair_dir.mkdir(parents=True, exist_ok=True)
    _write_image(pair_dir / "00000.png")
    _write_image(pair_dir / "00001.png")


def test_faceforensics_dataset_reads_images(tmp_path: Path):
    _build_fixture(tmp_path)
    transforms = build_transforms("xception")
    dataset = FaceForensicsDataset(tmp_path, split="train", manipulation="Deepfakes", compression="c23", num_frames=2, transform=transforms["train"])
    image, label = dataset[0]
    assert tuple(image.shape) == (3, 299, 299)
    assert int(label) in {0, 1}
    assert len(dataset) == 6


def test_build_balanced_sampler(tmp_path: Path):
    _build_fixture(tmp_path)
    dataset = FaceForensicsDataset(tmp_path, split="train", manipulation="Deepfakes", compression="c23", num_frames=2, transform=None)
    sampler = build_balanced_sampler(dataset)
    assert sampler.num_samples == len(dataset)
