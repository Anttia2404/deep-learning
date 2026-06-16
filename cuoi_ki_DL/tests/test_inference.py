from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from inference.face_detector import FaceDetectionConfig, FaceDetector


def test_haar_detector_returns_none_on_blank_image():
    detector = FaceDetector(FaceDetectionConfig(backend="haar", output_size=64))
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    face = detector.detect_and_crop(image)
    assert face is None


def test_local_checkpoint_roundtrip(tmp_path: Path):
    checkpoint_path = tmp_path / "dummy.pth"
    torch.save(
        {
            "model_state_dict": {
                "features.0.weight": torch.randn(8, 3, 3, 3)
            },
            "config": {"input_size": 256},
            "val_acc": 90.0,
        },
        checkpoint_path,
    )
    assert checkpoint_path.exists()


def test_write_dummy_video(tmp_path: Path):
    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"XVID"), 5, (32, 32))
    for _ in range(3):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert video_path.exists()
