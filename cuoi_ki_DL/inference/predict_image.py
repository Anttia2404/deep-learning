from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from datasets.transforms import build_transforms
from inference.face_detector import FaceDetectionConfig, FaceDetector
from models.model_factory import create_model
from training.trainer import resolve_device


def load_model_from_checkpoint(checkpoint_path: Path, model_name: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=config.get("num_classes", 2),
        dropout=config.get("dropout", 0.5),
        input_size=config.get("input_size", 299 if model_name == "xception" else 256),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_image_file(
    checkpoint_path: Path,
    input_path: Path,
    model_name: str = "xception",
    device: str | None = None,
    threshold: float = 0.5,
    detector_backend: str = "mtcnn",
):
    torch_device = resolve_device(device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, model_name, torch_device)
    input_size = checkpoint.get("config", {}).get("input_size", 299 if model_name == "xception" else 256)
    detector = FaceDetector(FaceDetectionConfig(backend=detector_backend, device=torch_device.type if torch_device.type == "cuda" else None, output_size=input_size))
    transforms = build_transforms(model_name)["test"]

    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {input_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = detector.detect_and_crop(rgb)
    if face is None:
        return {"label": "uncertain", "reason": "no face detected"}

    tensor = transforms(face).unsqueeze(0).to(torch_device)
    with torch.no_grad():
        logits = model(tensor)
        prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
    label = "FAKE" if prob_fake > threshold else "REAL"
    confidence = prob_fake if prob_fake > threshold else 1.0 - prob_fake
    return {
        "label": label,
        "prob_fake": prob_fake,
        "confidence": confidence,
        "checkpoint_val_acc": checkpoint.get("val_acc"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fake/real label for a single image")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", default="xception", choices=["xception", "mesonet", "mesoinception4"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--backend", default="mtcnn", choices=["mtcnn", "haar"])
    args = parser.parse_args()
    print(predict_image_file(args.checkpoint, args.input, args.model, args.device, args.threshold, args.backend))


if __name__ == "__main__":
    main()
