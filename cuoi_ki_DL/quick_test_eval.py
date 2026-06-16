import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import codebase elements
from datasets.transforms import build_transforms
from inference.face_detector import FaceDetectionConfig, FaceDetector
from inference.predict_image import load_model_from_checkpoint

# Define 10 test pairs from the official FaceForensics++ test split
TEST_PAIRS = [
    ['953', '974'], ['012', '026'], ['078', '955'], ['623', '630'], ['919', '015'],
    ['367', '371'], ['847', '906'], ['529', '633'], ['418', '507'], ['227', '169']
]

class LocalQuickTestDataset(Dataset):
    def __init__(self, data_root: Path, manipulation: str, compression: str, transform=None, num_frames: int = 16):
        self.data_root = data_root
        self.manipulation = manipulation
        self.compression = compression
        self.transform = transform
        self.num_frames = num_frames
        
        # Haar backend for quick CPU face extraction
        self.detector = FaceDetector(FaceDetectionConfig(backend="haar", output_size=299))
        self.samples = []
        
        # Build samples list (10 real and 10 fake videos)
        real_dir = data_root / "original_sequences" / "youtube" / compression / "videos"
        fake_dir = data_root / "manipulated_sequences" / manipulation / compression / "videos"
        
        for source, target in TEST_PAIRS:
            # 1. Real Video (Target)
            real_path = real_dir / f"{target}.mp4"
            if real_path.exists():
                self.samples.append((real_path, 0))
            
            # 2. Fake Video
            fake_path = fake_dir / f"{source}_{target}.mp4"
            if fake_path.exists():
                self.samples.append((fake_path, 1))

    def __len__(self):
        return len(self.samples)

    def _extract_face(self, frame: np.ndarray):
        return self.detector.detect_and_crop(frame)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            # Return dummy zero tensor if video can't be opened
            return torch.zeros(3, 299, 299), label
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return torch.zeros(3, 299, 299), label
            
        # Select evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        crops = []
        
        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop = self._extract_face(frame)
            if crop is not None:
                crops.append(crop)
                
        cap.release()
        
        if len(crops) == 0:
            # Fallback if no faces detected
            return torch.zeros(3, 299, 299), label
            
        # Select middle frame crop for prediction
        best_crop = crops[len(crops) // 2]
        
        if self.transform:
            best_crop = self.transform(best_crop)
            
        return best_crop, label

def evaluate_all():
    data_root = Path("d:/ffpp/data")
    ckpt_dir = Path("d:/ffpp/cuoi_ki_DL/checkpoints")
    models = {
        "Deepfakes": "best_Deepfakes_c23.pth",
        "Face2Face": "best_Face2Face_c23.pth",
        "FaceSwap": "best_FaceSwap_c23.pth",
        "NeuralTextures": "best_NeuralTextures_c23.pth"
    }
    
    device = torch.device("cpu")
    transforms = build_transforms("xception")
    
    results_summary = []
    
    print("="*60)
    print("STARTING LOCAL EVALUATION OF 4 MODELS ON TEST SUBSET")
    print("="*60)
    
    for manipulation, ckpt_name in models.items():
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"[-] Checkpoint not found: {ckpt_name}")
            continue
            
        print(f"[*] Evaluating {manipulation}...")
        t0 = time.time()
        
        # Load model
        model, _ = load_model_from_checkpoint(ckpt_path, "xception", device)
        model.eval()
        
        # Load dataset
        dataset = LocalQuickTestDataset(
            data_root=data_root,
            manipulation=manipulation,
            compression="c23",
            transform=transforms["test"],
            num_frames=16 # 16 frames for faster CPU prediction
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        probs = []
        labels = []
        
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
                probs.extend(batch_probs)
                labels.extend(lbls.numpy().tolist())
                
        y_true = np.array(labels)
        y_prob = np.array(probs)
        y_pred = (y_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred) * 100.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = float('nan')
            
        elapsed = time.time() - t0
        print(f"[+] Finished {manipulation} in {elapsed:.1f}s | Accuracy: {acc:.2f}% | F1: {f1:.4f}")
        
        results_summary.append({
            "Method": manipulation,
            "Accuracy": f"{acc:.2f}%",
            "F1-Score": f"{f1:.4f}",
            "Precision": f"{precision:.4f}",
            "Recall": f"{recall:.4f}",
            "AUC-ROC": f"{auc:.4f}"
        })
        
    # Generate Markdown Table
    print("\n" + "="*70)
    print("EVALUATION RESULTS TABLE (TEST SPLIT - 20 VIDEOS PER MODEL)")
    print("="*70)
    print("| Method | Accuracy | F1-Score | Precision | Recall | AUC-ROC |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results_summary:
        print(f"| **{r['Method']}** | {r['Accuracy']} | {r['F1-Score']} | {r['Precision']} | {r['Recall']} | {r['AUC-ROC']} |")
    print("="*70 + "\n")

if __name__ == "__main__":
    evaluate_all()
