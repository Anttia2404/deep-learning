from __future__ import annotations

import argparse
from pathlib import Path

from inference.predict_image import predict_image_file
from inference.predict_video import predict_video_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Local inference for FaceForensics++ checkpoints")
    parser.add_argument("checkpoint", type=Path, help="Path to .pth checkpoint")
    parser.add_argument("input", type=Path, help="Image or video path")
    parser.add_argument("--model", default="xception", choices=["xception", "mesonet", "mesoinception4"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--backend", default="mtcnn", choices=["mtcnn", "haar"])
    args = parser.parse_args()

    suffix = args.input.suffix.lower()
    if suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        result = predict_video_file(
            checkpoint_path=args.checkpoint,
            input_path=args.input,
            model_name=args.model,
            device=args.device,
            threshold=args.threshold,
            num_frames=args.num_frames,
            detector_backend=args.backend,
        )
    else:
        result = predict_image_file(
            checkpoint_path=args.checkpoint,
            input_path=args.input,
            model_name=args.model,
            device=args.device,
            threshold=args.threshold,
            detector_backend=args.backend,
        )

    print(result)


if __name__ == "__main__":
    main()
