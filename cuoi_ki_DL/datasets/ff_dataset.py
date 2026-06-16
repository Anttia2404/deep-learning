from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from inference.face_detector import FaceDetectionConfig, FaceDetector

MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


@dataclass
class SampleRecord:
    path: Path
    label: int
    video_id: str
    manipulation: str


class FaceForensicsDataset(Dataset):
    """Frame-level dataset backed by extracted face crops on disk."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        manipulation: str,
        compression: str = "c23",
        num_frames: int = 270,
        transform: Callable | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.manipulation = manipulation
        self.compression = compression
        self.num_frames = num_frames
        self.transform = transform
        self.samples: List[SampleRecord] = []
        self._build_dataset()

    def _load_split_ids(self) -> list[str]:
        split_path = self.data_root / "splits" / f"{self.split}.json"
        with split_path.open("r", encoding="utf-8") as handle:
            raw_ids = json.load(handle)
        normalized = []
        for entry in raw_ids:
            if isinstance(entry, list):
                normalized.extend(str(item).zfill(3) for item in entry)
            else:
                normalized.append(str(entry).zfill(3))
        return sorted(set(normalized))

    @staticmethod
    def _sample_frame_names(frame_names: Sequence[str], n: int) -> list[str]:
        if len(frame_names) <= n:
            return list(frame_names)
        indices = np.linspace(0, len(frame_names) - 1, n, dtype=int)
        return [frame_names[idx] for idx in indices]

    def _collect_real_samples(self, video_ids: Iterable[str]) -> None:
        real_dir = self.data_root / "original_sequences" / "youtube" / self.compression / "faces"
        for video_id in video_ids:
            face_dir = real_dir / video_id
            if not face_dir.exists():
                continue
            frames = sorted([f.name for f in face_dir.iterdir() if f.is_file()])
            for frame_name in self._sample_frame_names(frames, self.num_frames):
                self.samples.append(SampleRecord(face_dir / frame_name, 0, video_id, "original"))

    def _collect_fake_samples(self, video_ids: Iterable[str]) -> None:
        manipulations = MANIPULATIONS if self.manipulation == "all" else [self.manipulation]
        for manipulation in manipulations:
            fake_root = self.data_root / "manipulated_sequences" / manipulation / self.compression / "faces"
            if not fake_root.exists():
                continue
            for pair_dir in fake_root.iterdir():
                if not pair_dir.is_dir():
                    continue
                parts = pair_dir.name.split("_")
                if not parts:
                    continue
                if not any(part in video_ids for part in parts[:2]):
                    continue
                frames = sorted([f.name for f in pair_dir.iterdir() if f.is_file()])
                for frame_name in self._sample_frame_names(frames, self.num_frames):
                    self.samples.append(SampleRecord(pair_dir / frame_name, 1, pair_dir.name, manipulation))

    def _build_dataset(self) -> None:
        video_ids = self._load_split_ids()
        self._collect_real_samples(video_ids)
        self._collect_fake_samples(video_ids)
        if not self.samples:
            raise RuntimeError(
                f"No samples found for split={self.split}, manipulation={self.manipulation}, compression={self.compression}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = cv2.imread(str(sample.path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {sample.path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
        return image, torch.tensor(sample.label, dtype=torch.long)


class FaceForensicsVideoDataset(Dataset):
    """Video-level dataset that extracts representative face crops on-the-fly."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        manipulation: str,
        compression: str = "c23",
        num_frames: int = 32,
        transform: Callable | None = None,
        detector_backend: str = "mtcnn",
        detector_device: str | None = None,
        face_margin: float = 0.4,
        face_size: int = 299,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.manipulation = manipulation
        self.compression = compression
        self.num_frames = num_frames
        self.transform = transform
        self.detector = FaceDetector(
            FaceDetectionConfig(backend=detector_backend, device=detector_device, margin=face_margin, output_size=face_size)
        )
        self.samples: list[tuple[Path, int]] = []
        self._build_video_index()

    def _load_split_ids(self) -> list[str]:
        split_path = self.data_root / "splits" / f"{self.split}.json"
        with split_path.open("r", encoding="utf-8") as handle:
            raw_ids = json.load(handle)
        return sorted({str(item[0] if isinstance(item, list) else item).zfill(3) for item in raw_ids})

    def _build_video_index(self) -> None:
        video_ids = set(self._load_split_ids())
        real_dir = self.data_root / "original_sequences" / "youtube" / self.compression / "videos"
        for video_id in video_ids:
            path = real_dir / f"{video_id}.mp4"
            if path.exists():
                self.samples.append((path, 0))

        manipulations = MANIPULATIONS if self.manipulation == "all" else [self.manipulation]
        for manipulation in manipulations:
            fake_dir = self.data_root / "manipulated_sequences" / manipulation / self.compression / "videos"
            if not fake_dir.exists():
                continue
            for path in fake_dir.iterdir():
                if not path.is_file():
                    continue
                stem = path.stem.split("_")
                if stem and stem[0] in video_ids:
                    self.samples.append((path, 1))

        if not self.samples:
            raise RuntimeError("No video samples found for FaceForensicsVideoDataset")

    @staticmethod
    def _frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
        return np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)

    def _extract_face(self, frame: np.ndarray):
        face = self.detector.detect_and_crop(frame)
        if face is None:
            return None
        if self.transform is not None:
            return self.transform(face)
        tensor = torch.from_numpy(np.transpose(face, (2, 0, 1))).float() / 255.0
        return tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._frame_indices(total_frames, self.num_frames)
        crop = None
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            success, frame = cap.read()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_crop = self._extract_face(frame)
            if detected_crop is not None:
                crop = detected_crop
                break
        cap.release()

        if crop is None:
            fallback_shape = (3, self.detector.config.output_size, self.detector.config.output_size)
            return torch.zeros(fallback_shape, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        return crop, torch.tensor(label, dtype=torch.long)


def build_balanced_sampler(dataset: FaceForensicsDataset) -> WeightedRandomSampler:
    labels = np.array([sample.label for sample in dataset.samples])
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
