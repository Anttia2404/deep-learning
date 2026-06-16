from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    from facenet_pytorch import MTCNN
except Exception:  # pragma: no cover - optional dependency resolution
    MTCNN = None


@dataclass
class FaceDetectionConfig:
    backend: str = "mtcnn"
    device: str | None = None
    margin: float = 0.4
    output_size: int = 299
    haar_cascade_path: str | None = None


class FaceDetector:
    def __init__(self, config: FaceDetectionConfig | None = None) -> None:
        self.config = config or FaceDetectionConfig()
        backend = self.config.backend.lower()
        if backend == "mtcnn":
            if MTCNN is None:
                raise ImportError("facenet-pytorch is required for MTCNN backend")
            self.detector = MTCNN(
                image_size=self.config.output_size,
                margin=max(0, int(self.config.margin * self.config.output_size)),
                keep_all=False,
                device=self.config.device,
                post_process=False,
            )
        elif backend == "haar":
            cascade_path = self.config.haar_cascade_path or cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
        else:
            raise ValueError(f"Unsupported face detector backend: {self.config.backend}")

    def detect_and_crop(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        backend = self.config.backend.lower()
        if backend == "mtcnn":
            tensor = self.detector(frame_rgb)
            if tensor is None:
                return None
            face = tensor.permute(1, 2, 0).cpu().numpy()
            face = np.clip(face, 0, 255).astype(np.uint8)
            return face

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        margin_w = int(w * self.config.margin)
        margin_h = int(h * self.config.margin)
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(frame_rgb.shape[1], x + w + margin_w)
        y2 = min(frame_rgb.shape[0], y + h + margin_h)
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (self.config.output_size, self.config.output_size), interpolation=cv2.INTER_LINEAR)
