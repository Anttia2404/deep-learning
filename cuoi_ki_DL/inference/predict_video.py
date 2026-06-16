from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from datasets.transforms import build_transforms
from inference.face_detector import FaceDetectionConfig, FaceDetector
from inference.predict_image import load_model_from_checkpoint
from training.trainer import resolve_device


def predict_video_file(
    checkpoint_path: Path,
    input_path: Path,
    model_name: str = "xception",
    device: str | None = None,
    threshold: float = 0.5,
    num_frames: int = 32,
    detector_backend: str = "mtcnn",
    aggregation: str = "mean",
):
    torch_device = resolve_device(device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, model_name, torch_device)
    input_size = checkpoint.get("config", {}).get("input_size", 299 if model_name == "xception" else 256)
    detector = FaceDetector(FaceDetectionConfig(backend=detector_backend, device=torch_device.type if torch_device.type == "cuda" else None, output_size=input_size))
    transform = build_transforms(model_name)["test"]

    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    probs = []

    with torch.no_grad():
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            success, frame = cap.read()
            if not success:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = detector.detect_and_crop(rgb)
            if face is None:
                continue
            tensor = transform(face).unsqueeze(0).to(torch_device)
            logits = model(tensor)
            probs.append(torch.softmax(logits, dim=1)[0, 1].item())

    cap.release()
    if not probs:
        return {"label": "uncertain", "reason": "no face detected", "frames_analyzed": 0}

    if aggregation == "majority":
        score = float(np.mean([1 if prob > threshold else 0 for prob in probs]))
    else:
        score = float(np.mean(probs))

    label = "FAKE" if score > threshold else "REAL"
    confidence = score if score > threshold else 1.0 - score
    return {
        "label": label,
        "prob_fake": score,
        "confidence": confidence,
        "frames_analyzed": len(probs),
        "checkpoint_val_acc": checkpoint.get("val_acc"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fake/real label for a video")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", default="xception", choices=["xception", "mesonet", "mesoinception4"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--backend", default="mtcnn", choices=["mtcnn", "haar"])
    parser.add_argument("--aggregation", default="mean", choices=["mean", "majority"])
    args = parser.parse_args()
    print(
        predict_video_file(
            checkpoint_path=args.checkpoint,
            input_path=args.input,
            model_name=args.model,
            device=args.device,
            threshold=args.threshold,
            num_frames=args.num_frames,
            detector_backend=args.backend,
            aggregation=args.aggregation,
        )
    )


if __name__ == "__main__":
    main()
