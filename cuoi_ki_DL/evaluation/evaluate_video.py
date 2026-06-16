import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from datasets.ff_dataset import FaceForensicsVideoDataset
from datasets.transforms import build_transforms
from inference.predict_image import load_model_from_checkpoint
from training.trainer import resolve_device

def evaluate_video_level(
    checkpoint_path: Path,
    data_root: Path,
    model_name: str,
    manipulation: str,
    compression: str,
    num_frames: int = 32,
    detector_backend: str = "mtcnn",
    batch_size: int = 16,
    device_name: str | None = None
):
    device = resolve_device(device_name)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, model_name, device)
    transforms = build_transforms(model_name)
    
    # Sử dụng FaceForensicsVideoDataset vì trên Kaggle là file video .mp4
    dataset = FaceForensicsVideoDataset(
        data_root=data_root,
        split="test",
        manipulation=manipulation,
        compression=compression,
        num_frames=num_frames,
        transform=transforms["test"],
        detector_backend=detector_backend,
        detector_device="cuda" if device.type == "cuda" else None
    )
    
    # Đặt num_workers=0 để tránh lỗi CUDA fork trên Kaggle
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    probs = []
    labels = []
    
    print(f"🔄 Đang tiến hành đánh giá trên tập TEST ({len(dataset)} videos)...")
    
    with torch.no_grad():
        for i, (images, batch_labels) in enumerate(loader):
            images = images.to(device)
            logits = model(images)
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
            probs.extend(batch_probs)
            labels.extend(batch_labels.numpy().tolist())
            if (i+1) % 5 == 0 or (i+1) == len(loader):
                print(f"   Đã quét: {(i+1)*batch_size}/{len(dataset)} videos")
                
    y_true = np.array(labels)
    y_prob = np.array(probs)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Tính các chỉ số đánh giá
    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float('nan')
        
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    results = {
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc,
        "confusion_matrix": cm
    }
    
    # In ra dạng bảng Markdown cực đẹp cho báo cáo
    print("\n" + "="*50)
    print("📈 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (VIDEO-LEVEL)")
    print("="*50)
    print(f"Mô hình: {model_name.upper()} | Tập dữ liệu: {manipulation} (nén {compression})")
    print(f"Checkpoint tốt nhất tại: {checkpoint_path.name}")
    print("-"*50)
    print(f"| Chỉ số (Metric) | Giá trị (Value) |")
    print(f"| :--- | :--- |")
    print(f"| **Accuracy (Độ chính xác)** | {acc:.2f}% |")
    print(f"| **F1-Score** | {f1:.4f} |")
    print(f"| **Precision (Độ chính xác dự báo)** | {precision:.4f} |")
    print(f"| **Recall (Độ phủ)** | {recall:.4f} |")
    print(f"| **AUC-ROC** | {auc:.4f} |")
    print("-"*50)
    print(f"Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    print("="*50 + "\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on FaceForensics++ video-level test split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", default="xception")
    parser.add_argument("--manipulation", default="Deepfakes")
    parser.add_argument("--compression", default="c23")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--backend", default="mtcnn")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    
    evaluate_video_level(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        model_name=args.model,
        manipulation=args.manipulation,
        compression=args.compression,
        num_frames=args.num_frames,
        detector_backend=args.backend,
        batch_size=args.batch_size,
        device_name=args.device
    )

if __name__ == "__main__":
    main()
