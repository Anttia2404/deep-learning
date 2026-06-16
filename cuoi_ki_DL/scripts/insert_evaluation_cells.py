import json
from pathlib import Path

notebook_dir = Path(r'd:\ffpp\cuoi_ki_DL\notebooks')
notebook_paths = list(notebook_dir.glob('*.ipynb'))

markdown_cell = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 14 — Đánh giá chi tiết trên tập TEST (F1-Score, Accuracy, Precision, Recall) 📈\n",
    "\n",
    "Cell này sẽ chạy đánh giá mô hình tốt nhất (`best_*.pth`) vừa train xong trên tập kiểm thử độc lập (Test Set) của video-level, hiển thị chi tiết các chỉ số F1, Accuracy, Precision, Recall và Confusion Matrix cho báo cáo của bạn."
   ]
}

# The code will dynamically find the checkpoint based on files in /kaggle/working/checkpoints/
code_cell_template = """import os
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# Tự động import từ codebase đã copy
import sys
sys.path.append('/kaggle/working/ffpp')
from datasets.ff_dataset import FaceForensicsVideoDataset
from datasets.transforms import build_transforms
from inference.predict_image import load_model_from_checkpoint
from training.trainer import resolve_device

# Xác định tên manipulation của notebook này
manipulation = "{manipulation}"

best_ckpt_path = Path('/kaggle/working/checkpoints') / f'best_{{manipulation}}_c23.pth'

if best_ckpt_path.exists():
    device = resolve_device('cuda')
    model, checkpoint = load_model_from_checkpoint(best_ckpt_path, 'xception', device)
    transforms = build_transforms('xception')
    
    test_dataset = FaceForensicsVideoDataset(
        data_root=DATA_ROOT_FINAL,
        split="test",
        manipulation=manipulation,
        compression="c23",
        num_frames=32,
        transform=transforms["test"],
        detector_backend="mtcnn",
        detector_device="cuda"
    )
    
    # Đặt num_workers=0 để tránh lỗi multiprocessing CUDA
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    probs = []
    labels = []
    
    print(f"🔄 Đang đánh giá mô hình {{manipulation}} trên tập TEST ({{len(test_dataset)}} videos)...")
    with torch.no_grad():
        for i, (images, batch_labels) in enumerate(test_loader):
            images = images.to(device)
            logits = model(images)
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
            probs.extend(batch_probs)
            labels.extend(batch_labels.numpy().tolist())
            if (i+1) % 5 == 0 or (i+1) == len(test_loader):
                print(f"   Đã quét: {{min((i+1)*16, len(test_dataset))}}/{{len(test_dataset)}} videos")
                
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
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    print("\\n" + "="*50)
    print("📈 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (VIDEO-LEVEL)")
    print("="*50)
    print(f"| Chỉ số (Metric) | Giá trị (Value) |")
    print(f"| :--- | :--- |")
    print(f"| **Accuracy (Độ chính xác)** | {{acc:.2f}}% |")
    print(f"| **F1-Score** | {{f1:.4f}} |")
    print(f"| **Precision (Độ chính xác dự báo)** | {{precision:.4f}} |")
    print(f"| **Recall (Độ phủ)** | {{recall:.4f}} |")
    print(f"| **AUC-ROC** | {{auc:.4f}} |")
    print("-"*50)
    print(f"Confusion Matrix: TN={{cm[0][0]}}, FP={{cm[0][1]}}, FN={{cm[1][0]}}, TP={{cm[1][1]}}")
    print("="*50 + "\\n")
else:
    print(f"❌ Không tìm thấy file checkpoint: {{best_ckpt_path}}")
"""

for nb_path in notebook_paths:
    print(f'Processing {nb_path.name}...')
    with nb_path.open('r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Determine manipulation based on file name
    if 'deepfakes' in nb_path.name:
        manip = 'Deepfakes'
    elif 'face2face' in nb_path.name:
        manip = 'Face2Face'
    elif 'faceswap' in nb_path.name:
        manip = 'FaceSwap'
    elif 'neuraltextures' in nb_path.name:
        manip = 'NeuralTextures'
    else:
        manip = 'Deepfakes'
        
    # Check if already has Cell 14
    already_has = False
    for cell in nb['cells']:
        if any('Cell 14 — Đánh giá chi tiết trên tập TEST' in line for line in cell.get('source', [])):
            already_has = True
            break
            
    if not already_has:
        code_source = [line + '\n' for line in code_cell_template.format(manipulation=manip).split('\n')]
        # Strip trailing newlines from array elements to prevent double spacing in ipynb formatting
        code_source = [line for line in code_source]
        
        # We also need to strip out empty strings at the end
        if code_source and code_source[-1] == '\n':
            code_source.pop()
            
        nb['cells'].append(markdown_cell)
        nb['cells'].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_source
        })
        
        with nb_path.open('w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f'  -> Appended evaluation cells to {nb_path.name}')
    else:
        print('  -> Evaluation cells already present')

print('Done!')
