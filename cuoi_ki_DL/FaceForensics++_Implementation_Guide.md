# FaceForensics++ — Tài Liệu Kỹ Thuật Triển Khai Toàn Diện

> **Dành cho AI Coding Agent**: Tài liệu này đủ để tái hiện toàn bộ paper từ đầu mà không cần đọc paper gốc.  
> Paper gốc: *"FaceForensics++: Learning to Detect Manipulated Facial Images"* — Rössler et al., ICCV 2019.
>
> **Workflow khuyên dùng**: Train trên **Kaggle** (GPU T4 miễn phí, 12h/session) → Download checkpoint (~90MB) → Inference trên **local**. Không cần máy local có GPU.

---

## Mục Lục

1. [Tổng Quan & Đóng Góp](#1-tổng-quan--đóng-góp)
2. [Kiến Trúc Mô Hình](#2-kiến-trúc-mô-hình)
3. [Dataset](#3-dataset)
4. [Pipeline Tiền Xử Lý](#4-pipeline-tiền-xử-lý)
5. [Hyperparameters & Training Strategy](#5-hyperparameters--training-strategy)
6. [Loss Functions & Optimizer](#6-loss-functions--optimizer)
7. [Training Pipeline Chi Tiết](#7-training-pipeline-chi-tiết)
8. [Inference Pipeline](#8-inference-pipeline)
9. [Công Thức Toán Học & Hiện Thực Hóa](#9-công-thức-toán-học--hiện-thực-hóa)
10. [Cấu Trúc Repository](#10-cấu-trúc-repository)
11. [Roadmap Triển Khai Từng Bước (Kaggle → Local)](#11-roadmap-triển-khai-từng-bước-kaggle--local)
12. [Đánh Giá & Benchmark](#12-đánh-giá--benchmark)
13. [Rủi Ro, Điểm Mơ Hồ & Cách Xử Lý](#13-rủi-ro-điểm-mơ-hồ--cách-xử-lý)
14. [Snippets Code Tham Khảo](#14-snippets-code-tham-khảo)
15. [Local Inference Sau Khi Download Checkpoint](#15-local-inference-sau-khi-download-checkpoint)

---

## 1. Tổng Quan & Đóng Góp

### 1.1 Mục Tiêu Paper

FaceForensics++ giải quyết bài toán **phát hiện ảnh/video khuôn mặt đã bị thao túng (deepfake detection)**. Paper có hai đóng góp lớn:

1. **Dataset benchmark chuẩn**: Xây dựng dataset quy mô lớn gồm 1.000 video gốc + các video giả mạo từ **4 phương pháp manipulation** khác nhau, ở **3 mức độ nén** (quality levels).
2. **Benchmark phát hiện**: Đánh giá nhiều phương pháp phát hiện (cả truyền thống lẫn deep learning) trên dataset này, cho thấy mạng học sâu vượt trội nhưng vẫn gặp khó khăn ở video nén mạnh.

### 1.2 Bốn Phương Pháp Manipulation

| ID | Tên | Loại | Mô Tả |
|----|-----|------|--------|
| DF | **Deepfakes** | Identity swap (GAN-based) | Hoán đổi khuôn mặt bằng autoencoder-GAN, lấy từ các video Deepfakes trên internet |
| F2F | **Face2Face** | Expression transfer | Chuyển biểu cảm từ video nguồn sang video đích (retargeting) |
| FS | **FaceSwap** | Identity swap (graphic-based) | Hoán đổi khuôn mặt theo phương pháp đồ họa máy tính (không dùng GAN) |
| NT | **NeuralTextures** | Expression transfer (neural) | Dùng neural rendering để tổng hợp khuôn mặt mới từ neural texture |

### 1.3 Ý Tưởng Cốt Lõi

- Dùng **XceptionNet** (fine-tuned từ pretrained ImageNet) làm backbone chính cho phát hiện.
- Phát hiện ở **frame level** (mỗi frame là 1 sample), kết quả video-level là trung bình hoặc majority vote.
- So sánh với các baseline: MesoNet, XceptionNet, FaceWarping artifacts detection.
- Chứng minh rằng **compression (H.264)** là thách thức lớn nhất với các detector.

---

## 2. Kiến Trúc Mô Hình

### 2.1 Model Chính: XceptionNet (Binary Classifier)

Paper sử dụng **XceptionNet** được fine-tune cho bài toán phân loại nhị phân (real vs. fake).

```
Input: RGB frame 299×299×3
  │
  ▼
XceptionNet Backbone (ImageNet pretrained)
  ├── Entry Flow
  │     ├── Conv2D(32, 3×3, stride=2) → BN → ReLU
  │     ├── Conv2D(64, 3×3) → BN → ReLU
  │     └── 3× SeparableConv Blocks (128→256→728 channels)
  │
  ├── Middle Flow
  │     └── 8× SeparableConv Residual Blocks (728 channels)
  │
  └── Exit Flow
        ├── SeparableConv Block (728→1024)
        ├── SeparableConv(1536) → BN → ReLU
        ├── SeparableConv(2048) → BN → ReLU
        ├── GlobalAveragePooling2D
        ├── Dropout(p=0.5)  ← THÊM VÀO khi fine-tune
        └── FC(2) → Softmax  ← THAY THẾ head gốc

Output: [P(real), P(fake)]
```

**Lưu ý fine-tuning**:
- Thay FC head cuối (từ 1000 classes → 2 classes)
- Thêm Dropout(0.5) trước FC
- Toàn bộ backbone được **unfreeze** và fine-tuned (không chỉ fine-tune head)

### 2.2 Baseline Models

#### MesoNet (Afchar et al., 2018)

```
Input: 256×256×3
  ├── Conv2D(8, 3×3) → BN → ReLU → MaxPool(2×2)
  ├── Conv2D(8, 5×5) → BN → ReLU → MaxPool(2×2)
  ├── Conv2D(16, 5×5) → BN → ReLU → MaxPool(4×4)
  ├── Conv2D(16, 5×5) → BN → ReLU → MaxPool(4×4)
  ├── Flatten
  ├── Dropout(0.5)
  ├── FC(16) → LeakyReLU
  ├── Dropout(0.5)
  └── FC(1) → Sigmoid
```

#### MesoInception4 (cải tiến của MesoNet)

Thay 2 conv đầu bằng **Inception modules**, giữ nguyên phần còn lại của MesoNet.

Inception module:
```
Input →
  ├── Branch1: Conv(1×1)
  ├── Branch2: Conv(1×1) → Conv(3×3)
  ├── Branch3: Conv(1×1) → Conv(3×3) → Conv(3×3)
  └── Branch4: AvgPool(3×3, stride=1) → Conv(1×1)
Concatenate branches → Output
```

### 2.3 Face X-ray (Phương Pháp Phụ - Section 5 của paper)

Paper cũng đề cập đến phương pháp dựa trên **blending boundary artifacts**:
- Phát hiện "đường biên pha trộn" (blending boundary) ở khuôn mặt
- Output thêm **face X-ray map** (segmentation mask của boundary)
- Đây là cách tiếp cận bổ sung, không phải model chính

---

## 3. Dataset

### 3.1 Thống Kê Dataset

| Thành phần | Số lượng |
|-----------|----------|
| Video gốc (real) | 1.000 |
| Video DF (Deepfakes) | 1.000 |
| Video F2F (Face2Face) | 1.000 |
| Video FS (FaceSwap) | 1.000 |
| Video NT (NeuralTextures) | 1.000 |
| **Tổng video** | **5.000** |
| Frames/video (trung bình) | ~300 |
| **Tổng frames (ước tính)** | **~1.5 triệu** |

### 3.2 Nguồn Video Gốc

- **1.000 video YouTube**: Lấy từ 977 diễn viên khác nhau trong các video YouTube có Creative Commons license.
- Điều kiện: khuôn mặt nhìn thẳng, không bị che khuất, đủ ánh sáng.
- Cắt các đoạn liên tục chứa khuôn mặt.

### 3.3 Ba Mức Độ Nén (Compression)

| Level | Codec | CRF/Quality | Ký hiệu |
|-------|-------|-------------|---------|
| RAW | Không nén | Lossless | c0 |
| High Quality | H.264 | CRF=23 | c23 |
| Low Quality | H.264 | CRF=40 | c40 |

> **Quan trọng**: CRF (Constant Rate Factor) càng cao → nén càng mạnh → chất lượng càng thấp.

### 3.4 Chia Tập Dữ Liệu

Tỷ lệ **720 / 140 / 140** (Train / Val / Test) — áp dụng cho tất cả 5 categories:

```
Tổng: 1000 videos/category
  ├── Train: 720 videos/category
  ├── Val:   140 videos/category
  └── Test:  140 videos/category
```

Khi train binary classifier (real vs. fake):
- Real: 720 real videos
- Fake: 720 fake videos của từng manipulation method (train riêng hoặc gộp)

### 3.5 Cách Thu Thập Data Deepfakes (DF)

Paper không tự tạo deepfakes mà **crawl từ internet**:
- Tìm các cặp video (nguồn, đích) liên quan đến cùng một cặp người
- Sử dụng script để match video gốc trong dataset với video deepfake tìm được
- Đây là lý do DF có chất lượng thao túng đa dạng nhất

---

## 4. Pipeline Tiền Xử Lý

### 4.1 Face Detection & Tracking

```python
# Bước 1: Phát hiện khuôn mặt trên từng frame
# Paper dùng dlib's frontal face detector + shape predictor 68 landmarks

import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Với mỗi frame:
faces = detector(frame_gray, 1)
for face in faces:
    landmarks = predictor(frame_gray, face)
```

### 4.2 Face Extraction & Cropping

```python
import cv2
import numpy as np

def extract_face(frame, landmarks, margin=0.4):
    """
    Crop khuôn mặt với margin mở rộng.
    margin=0.4 nghĩa là mở rộng 40% về mỗi phía.
    """
    x_coords = [landmarks.part(i).x for i in range(68)]
    y_coords = [landmarks.part(i).y for i in range(68)]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    w = x_max - x_min
    h = y_max - y_min
    
    # Mở rộng bounding box
    x_min = max(0, int(x_min - margin * w))
    x_max = min(frame.shape[1], int(x_max + margin * w))
    y_min = max(0, int(y_min - margin * h))
    y_max = min(frame.shape[0], int(y_max + margin * h))
    
    face_crop = frame[y_min:y_max, x_min:x_max]
    
    # Resize về kích thước chuẩn
    face_resized = cv2.resize(face_crop, (299, 299))  # XceptionNet input size
    
    return face_resized
```

### 4.3 Frame Sampling Strategy

Paper không dùng toàn bộ frames mà **sample** để tránh quá nhiều redundancy:

```python
def sample_frames(video_path, num_frames=270, method='uniform'):
    """
    Sample frames từ video.
    Paper dùng uniform sampling trong training.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if method == 'uniform':
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    elif method == 'random':
        indices = sorted(np.random.choice(total_frames, num_frames, replace=False))
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames
```

**Lưu ý thực tế**: Paper đề cập train với **~270 frames/video** (khoảng 1 frame/giây với video 25fps, 10 giây). Tuy nhiên chi tiết chính xác không được nêu rõ — xem Section 13 để xử lý.

### 4.4 Data Augmentation

```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),          # XceptionNet input
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],  # Paper normalize về [-1, 1]
        std=[0.5, 0.5, 0.5]
    )
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

**Normalization**: Paper dùng `[-1, 1]` range (mean=0.5, std=0.5 cho từng channel), **không** dùng ImageNet stats.

---

## 5. Hyperparameters & Training Strategy

### 5.1 XceptionNet Hyperparameters

```yaml
# Toàn bộ hyperparameters từ paper + suy luận hợp lý

model:
  architecture: xceptionnet
  pretrained: imagenet
  input_size: [299, 299, 3]
  num_classes: 2
  dropout_rate: 0.5

training:
  optimizer: adam
  initial_lr: 0.0002          # 2e-4
  lr_scheduler: step_lr
  lr_decay_factor: 0.1
  lr_decay_epochs: [30, 50]   # Suy luận (paper không nêu rõ)
  weight_decay: 0.0           # Paper không đề cập
  batch_size: 32              # Suy luận (typical cho XceptionNet)
  num_epochs: 30              # Suy luận (early stopping)
  
data:
  frame_size: 299
  num_frames_train: 270       # ~1 frame/giây
  num_frames_test: 270
  compression: [c0, c23, c40] # Train/eval trên từng level riêng
  
  split:
    train: 720
    val: 140
    test: 140
```

### 5.2 MesoNet Hyperparameters

```yaml
model:
  architecture: mesoinception4
  input_size: [256, 256, 3]
  num_classes: 1              # Binary output với sigmoid

training:
  optimizer: adam
  initial_lr: 0.001           # 1e-3 (từ paper MesoNet gốc)
  lr_scheduler: reduce_on_plateau
  lr_patience: 5
  batch_size: 75
  num_epochs: 30
```

### 5.3 Training per Manipulation Method

Paper train **riêng biệt** cho từng manipulation method:
- Model_DF: train trên real vs. Deepfakes
- Model_F2F: train trên real vs. Face2Face
- Model_FS: train trên real vs. FaceSwap
- Model_NT: train trên real vs. NeuralTextures
- Model_ALL: train trên real vs. tất cả fake (gộp lại)

---

## 6. Loss Functions & Optimizer

### 6.1 Loss Function

```python
import torch
import torch.nn as nn

# Binary Cross-Entropy với logits (XceptionNet: 2 outputs + CrossEntropy)
criterion = nn.CrossEntropyLoss()

# Hoặc dùng BCEWithLogitsLoss nếu dùng 1 output
# criterion = nn.BCEWithLogitsLoss()

# Với label smoothing (tùy chọn, không được đề cập trong paper):
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 6.2 Optimizer

```python
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999),    # PyTorch defaults = Adam defaults
    eps=1e-8,
    weight_decay=0.0
)
```

### 6.3 Learning Rate Scheduler

```python
# Paper đề cập giảm LR nhưng không nêu chi tiết strategy
# Suy luận hợp lý: StepLR hoặc ReduceLROnPlateau

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,      # Giảm mỗi 10 epochs
    gamma=0.1          # Giảm 10x
)

# Hoặc:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=5,
    verbose=True
)
```

---

## 7. Training Pipeline Chi Tiết

### 7.1 Dataset Class

```python
import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class FaceForensicsDataset(Dataset):
    def __init__(
        self, 
        data_root,           # /path/to/FaceForensics++/
        split,               # 'train', 'val', 'test'
        manipulation,        # 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'all'
        compression,         # 'c0', 'c23', 'c40'
        num_frames=270,
        transform=None
    ):
        self.data_root = data_root
        self.split = split
        self.manipulation = manipulation
        self.compression = compression
        self.num_frames = num_frames
        self.transform = transform
        
        self.samples = []  # List of (frame_path, label)
        self._build_dataset()
    
    def _build_dataset(self):
        """Build danh sách tất cả (face_image_path, label) pairs."""
        split_file = os.path.join(
            self.data_root, 'splits', f'{self.split}.json'
        )
        import json
        with open(split_file) as f:
            video_ids = json.load(f)
        
        # Real samples
        real_dir = os.path.join(
            self.data_root, 'original_sequences', 
            'youtube', self.compression, 'faces'
        )
        for vid_id in video_ids:
            face_dir = os.path.join(real_dir, vid_id)
            if not os.path.exists(face_dir):
                continue
            frames = sorted(os.listdir(face_dir))
            # Sample frames uniformly
            sampled = self._sample_frames(frames, self.num_frames)
            for f in sampled:
                self.samples.append((os.path.join(face_dir, f), 0))  # label=0: real
        
        # Fake samples
        manips = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'] \
                 if self.manipulation == 'all' else [self.manipulation]
        
        for manip in manips:
            fake_dir = os.path.join(
                self.data_root, 'manipulated_sequences',
                manip, self.compression, 'faces'
            )
            for vid_id in video_ids:
                # Video pairs: "001_003" = source 001, target 003
                for vid_pair in os.listdir(fake_dir):
                    if vid_id in vid_pair:
                        face_dir = os.path.join(fake_dir, vid_pair)
                        if not os.path.exists(face_dir):
                            continue
                        frames = sorted(os.listdir(face_dir))
                        sampled = self._sample_frames(frames, self.num_frames)
                        for fr in sampled:
                            self.samples.append(
                                (os.path.join(face_dir, fr), 1)  # label=1: fake
                            )
    
    def _sample_frames(self, frames, n):
        """Uniform sampling."""
        if len(frames) <= n:
            return frames
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i] for i in indices]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)
```

### 7.2 Model Definition

```python
import torch
import torch.nn as nn
from torchvision.models import inception_v3
# Dùng pretrainedmodels hoặc timm cho XceptionNet

def get_xception(pretrained=True, num_classes=2, dropout=0.5):
    """
    Load XceptionNet. Dùng timm library là cách đơn giản nhất.
    pip install timm
    """
    import timm
    model = timm.create_model('xception', pretrained=pretrained)
    
    # Lấy số features ở layer cuối
    num_features = model.fc.in_features
    
    # Thay head
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_features, num_classes)
    )
    
    return model


class MesoInception4(nn.Module):
    """
    Hiện thực MesoInception4 từ Afchar et al. 2018.
    """
    def __init__(self):
        super().__init__()
        
        self.inception_block1 = InceptionBlock(3, 8)
        self.inception_block2 = InceptionBlock(8, 8)
        
        self.conv1 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(4, 4)
        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 4 * 4, 16)   # Sau 3 lần pool 2x2 + 1 lần pool 4x4: 256/(2*2*4) = 8... tính lại bên dưới
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Tính spatial size sau pooling:
        # Input: 256×256
        # After inception_block1 + MaxPool(2,2): 128×128
        # After inception_block2 + MaxPool(2,2): 64×64
        # After conv1 + MaxPool(4,4): 16×16
        # After conv2 + MaxPool(4,4): 4×4
        # → fc1 input: 16 * 4 * 4 = 256
    
    def forward(self, x):
        x = self.inception_block1(x)
        x = self.inception_block2(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 4 branches, mỗi branch cho out_channels//4 channels
        # Tổng = out_channels
        branch_channels = out_channels // 4
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.ReLU()
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.pool(out)
        out = self.bn(out)
        return out
```

### 7.3 Training Loop

```python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc='Training')):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(fake)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_probs, all_labels


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    train_dataset = FaceForensicsDataset(
        data_root=config['data_root'],
        split='train',
        manipulation=config['manipulation'],
        compression=config['compression'],
        transform=train_transforms
    )
    val_dataset = FaceForensicsDataset(
        data_root=config['data_root'],
        split='val',
        manipulation=config['manipulation'],
        compression=config['compression'],
        transform=val_transforms
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = get_xception(pretrained=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    best_val_acc = 0.0
    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f"checkpoints/best_{config['manipulation']}_{config['compression']}.pth")
    
    return model
```

---

## 8. Inference Pipeline

### 8.1 Frame-level Prediction

```python
def predict_frame(model, frame_path, transform, device):
    """Dự đoán cho một frame."""
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        prob_fake = torch.softmax(output, dim=1)[0, 1].item()
    
    return prob_fake  # 0.0 = real, 1.0 = fake
```

### 8.2 Video-level Prediction (Aggregation)

```python
def predict_video(model, video_path, face_detector, transform, device, 
                  num_frames=100, threshold=0.5, method='mean'):
    """
    Dự đoán cho toàn bộ video.
    method: 'mean' (trung bình probabilities) hoặc 'majority' (vote)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frame_probs = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Face detection
        face = detect_and_crop_face(frame, face_detector)
        if face is None:
            continue
        
        # Prediction
        prob = predict_frame_from_array(model, face, transform, device)
        frame_probs.append(prob)
    
    cap.release()
    
    if not frame_probs:
        return 0.5, 'uncertain'
    
    if method == 'mean':
        final_prob = np.mean(frame_probs)
    elif method == 'majority':
        votes = [1 if p > threshold else 0 for p in frame_probs]
        final_prob = np.mean(votes)
    
    prediction = 'fake' if final_prob > threshold else 'real'
    return final_prob, prediction
```

### 8.3 Batch Evaluation Script

```python
def evaluate_on_test_set(model_path, data_root, manipulation, compression, device):
    """
    Đánh giá model trên test set, tính accuracy theo paper.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    model = get_xception(pretrained=False, num_classes=2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    test_dataset = FaceForensicsDataset(
        data_root=data_root,
        split='test',
        manipulation=manipulation,
        compression=compression,
        transform=val_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    _, acc, probs, labels = validate(model, test_loader, nn.CrossEntropyLoss(), device)
    auc = roc_auc_score(labels, probs)
    
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Test AUC-ROC: {auc:.4f}")
    
    return acc, auc
```

---

## 9. Công Thức Toán Học & Hiện Thực Hóa

### 9.1 Binary Cross-Entropy Loss

$$\mathcal{L}_{BCE} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i) \right]$$

```python
# Với 2 outputs + CrossEntropyLoss (equivalent):
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)  # labels: 0=real, 1=fake

# Với 1 output + BCEWithLogitsLoss:
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits.squeeze(), labels.float())
```

### 9.2 Video-level Score Aggregation

$$\hat{y}_{video} = \frac{1}{T} \sum_{t=1}^{T} P(\text{fake} | \mathbf{x}_t)$$

```python
video_score = np.mean([predict_frame(frame) for frame in sampled_frames])
prediction = 1 if video_score > 0.5 else 0
```

### 9.3 Accuracy Metric

$$\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$$

```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_pred)
```

### 9.4 Depthwise Separable Convolution (Xception building block)

$$\text{DSConv}(\mathbf{X}) = \text{Pointwise}(\text{Depthwise}(\mathbf{X}))$$

- **Depthwise**: Mỗi channel được convolved độc lập với 1 filter riêng
- **Pointwise**: Conv 1×1 để combine channels

```python
# PyTorch: groups=in_channels → depthwise
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, padding=padding, groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))
```

### 9.5 Xception Residual Block

```python
class XceptionBlock(nn.Module):
    def __init__(self, in_filters, out_filters, num_reps=3, 
                 stride=1, start_with_relu=True, grow_first=True):
        super().__init__()
        
        if out_filters != in_filters or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_filters)
            )
        else:
            self.skip = None
        
        rep = []
        filters = in_filters
        
        if grow_first:
            if start_with_relu:
                rep.append(nn.ReLU(inplace=True))
            rep.extend([
                SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_filters)
            ])
            filters = out_filters
        
        for i in range(num_reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.extend([
                SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(filters)
            ])
        
        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.extend([
                SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_filters)
            ])
        
        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        
        self.rep = nn.Sequential(*rep)
    
    def forward(self, inp):
        x = self.rep(inp)
        
        if self.skip is not None:
            skip = self.skip(inp)
        else:
            skip = inp
        
        x += skip
        return x
```

---

## 10. Cấu Trúc Repository

```
faceforensics-plus-plus/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│   ├── download_dataset.py          # Script download FF++ dataset chính thức
│   ├── extract_faces.py             # Chạy face detection + crop
│   ├── preprocess_videos.py         # Extract frames, apply compression
│   └── splits/
│       ├── train.json               # List video IDs cho train split
│       ├── val.json
│       └── test.json
│
├── models/
│   ├── __init__.py
│   ├── xception.py                  # XceptionNet definition + fine-tune logic
│   ├── meso_net.py                  # MesoNet & MesoInception4
│   └── model_factory.py             # Factory function: get_model(name)
│
├── datasets/
│   ├── __init__.py
│   ├── ff_dataset.py                # FaceForensicsDataset class
│   └── transforms.py                # train/val/test transforms
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                   # Training loop, validation
│   ├── losses.py                    # Loss functions
│   └── schedulers.py                # LR schedulers
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # Accuracy, AUC, F1, Confusion Matrix
│   ├── evaluate.py                  # Full evaluation script
│   └── visualize.py                 # Grad-CAM, ROC curves, confusion matrix plots
│
├── inference/
│   ├── __init__.py
│   ├── predict_image.py             # Single image prediction
│   ├── predict_video.py             # Video-level prediction với aggregation
│   └── face_detector.py             # dlib/RetinaFace face detection wrapper
│
├── configs/
│   ├── xception_c0.yaml             # Config cho từng combination
│   ├── xception_c23.yaml
│   ├── xception_c40.yaml
│   ├── meso_c0.yaml
│   └── base_config.yaml
│
├── checkpoints/                     # Saved model weights
│   └── .gitkeep
│
├── scripts/
│   ├── train_all.sh                 # Script train tất cả combinations
│   ├── eval_all.sh                  # Script eval tất cả combinations
│   └── download_pretrained.sh       # Download pretrained weights
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_visualization.ipynb
│   └── 03_results_analysis.ipynb
│
└── tests/
    ├── test_dataset.py
    ├── test_models.py
    └── test_inference.py
```

---

## 11. Roadmap Triển Khai Từng Bước (Kaggle → Local)

### Tổng Quan Workflow

```
[Bước 1] Đăng ký dataset FF++ → nhận link download
[Bước 2] Upload dataset lên Kaggle Dataset (1 lần duy nhất)
[Bước 3] Tạo Kaggle Notebook → train XceptionNet (~6-8h trên T4)
[Bước 4] Download checkpoint .pth về máy local (~90MB)
[Bước 5] Chạy inference local — không cần GPU
```

---

### Phase 0: Đăng Ký Dataset (Ngày 1)

```
1. Truy cập: https://github.com/ondyari/FaceForensics
2. Điền form xin quyền truy cập:
   https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform
3. Chờ email chứa download link (thường 1-2 ngày)
4. Sau khi nhận link, chạy official download script:
   https://github.com/ondyari/FaceForensics/blob/master/dataset/download_FaceForensics.py
```

**Chỉ tải c23 để bắt đầu** — đây là setting chính của paper và nhẹ nhất (~35GB so với c0 ~300GB):

```bash
# Chạy trên máy local, tải về ổ cứng ngoài hoặc thẳng lên Kaggle
python download_FaceForensics.py \
    /path/to/output \
    -d all \          # Tải tất cả: original + 4 manipulations
    -c c23 \          # Chỉ compression c23
    -t videos         # Tải dạng video (không phải frames)
```

> **Ước tính dung lượng c23**:
> - original: ~7GB
> - Deepfakes: ~7GB
> - Face2Face: ~7GB
> - FaceSwap: ~7GB
> - NeuralTextures: ~7GB
> - **Tổng: ~35GB**

---

### Phase 1: Upload Dataset Lên Kaggle (Ngày 2-3)

```
1. Đăng nhập Kaggle → "Datasets" → "New Dataset"
2. Tên dataset: "faceforensics-plus-plus-c23"
3. Upload toàn bộ thư mục c23 (có thể zip từng phần)
4. Visibility: Private
5. Chờ Kaggle xử lý (~vài giờ với 35GB)
```

**Cấu trúc thư mục upload lên Kaggle:**
```
faceforensics-plus-plus-c23/
├── original_sequences/youtube/c23/videos/
│   ├── 000.mp4
│   ├── 001.mp4
│   └── ... (1000 videos)
├── manipulated_sequences/
│   ├── Deepfakes/c23/videos/
│   │   ├── 000_003.mp4
│   │   └── ... (1000 videos)
│   ├── Face2Face/c23/videos/
│   ├── FaceSwap/c23/videos/
│   └── NeuralTextures/c23/videos/
└── splits/
    ├── train.json
    ├── val.json
    └── test.json
```

> **Tip**: Nếu upload chậm, dùng [Kaggle API](https://github.com/Kaggle/kaggle-api):
> ```bash
> kaggle datasets create -p /path/to/faceforensics-plus-plus-c23
> ```

---

### Phase 2: Setup Kaggle Notebook (Ngày 3)

```
1. Kaggle → "Notebooks" → "New Notebook"
2. Settings (bên phải):
   - Accelerator: GPU T4 x1
   - Persistence: Variables and Files
   - Internet: On (để pip install)
3. Add Dataset: tìm "faceforensics-plus-plus-c23" (dataset vừa upload)
4. Dataset sẽ mount tại: /kaggle/input/faceforensics-plus-plus-c23/
```

**Cell đầu tiên — Install dependencies:**
```python
# Kaggle đã có sẵn: torch, torchvision, opencv, numpy, sklearn
# Chỉ cần cài thêm:
!pip install timm -q          # XceptionNet pretrained
!pip install dlib -q          # Face detection (có thể chậm, dùng alternative bên dưới)

# Nếu dlib build lâu, dùng MTCNN thay thế (nhanh hơn):
!pip install facenet-pytorch -q
```

**Paths trong Kaggle Notebook:**
```python
# THAY ĐỔI: Không dùng /data/FaceForensics++ nữa
DATA_ROOT = '/kaggle/input/faceforensics-plus-plus-c23'
CHECKPOINT_DIR = '/kaggle/working/checkpoints'  # Lưu model ở đây
FACES_DIR = '/kaggle/working/faces'             # Face crops tạm thời

import os
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)
```

---

### Phase 3: Face Extraction Trên Kaggle (Ngày 3-4)

> **Lưu ý Kaggle**: `/kaggle/working` chỉ có 20GB. Face crops của toàn bộ dataset (~1.5M ảnh) cần ~30-50GB.  
> **Giải pháp**: Extract faces on-the-fly trong DataLoader thay vì lưu trước ra disk.

```python
# datasets/ff_dataset_kaggle.py
# Đọc video → extract face → trả về tensor, KHÔNG lưu ra disk

from facenet_pytorch import MTCNN
import torch
from torch.utils.data import Dataset
import cv2, json, os
import numpy as np

class FaceForensicsDatasetKaggle(Dataset):
    """
    Version tối ưu cho Kaggle: extract face on-the-fly từ video.
    Không cần pre-extract và lưu faces ra disk.
    """
    def __init__(self, data_root, split, manipulation, compression='c23',
                 num_frames=32, transform=None, device='cuda'):
        self.data_root = data_root
        self.num_frames = num_frames
        self.transform = transform
        
        # MTCNN chạy trên GPU — nhanh hơn dlib nhiều
        self.mtcnn = MTCNN(
            image_size=299, margin=40,
            keep_all=False, device=device
        )
        
        # Load split
        with open(os.path.join(data_root, 'splits', f'{split}.json')) as f:
            video_ids = [str(v[0]).zfill(3) for v in json.load(f)]
        
        self.samples = []  # List of (video_path, label)
        
        # Real videos
        real_dir = os.path.join(data_root, 'original_sequences', 
                                'youtube', compression, 'videos')
        for vid_id in video_ids:
            path = os.path.join(real_dir, f'{vid_id}.mp4')
            if os.path.exists(path):
                self.samples.append((path, 0))
        
        # Fake videos
        manips = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'] \
                 if manipulation == 'all' else [manipulation]
        
        for manip in manips:
            fake_dir = os.path.join(data_root, 'manipulated_sequences',
                                    manip, compression, 'videos')
            if not os.path.exists(fake_dir):
                continue
            for fname in os.listdir(fake_dir):
                # Lấy video có source hoặc target trong split
                src = fname.split('_')[0]
                if src in video_ids:
                    self.samples.append(
                        (os.path.join(fake_dir, fname), 1)
                    )
    
    def _extract_face_from_video(self, video_path):
        """Sample num_frames từ video, trả về list faces (tensor)."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return None
        
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        faces = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MTCNN detect + crop + resize về 299×299
            face = self.mtcnn(frame_rgb)  # Returns tensor (3,299,299) or None
            if face is not None:
                faces.append(face)
        
        cap.release()
        return faces if faces else None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        faces = self._extract_face_from_video(video_path)
        
        if faces is None or len(faces) == 0:
            # Fallback: trả về tensor zeros nếu không detect được face
            return torch.zeros(3, 299, 299), torch.tensor(label)
        
        # Lấy 1 frame ngẫu nhiên trong training, frame giữa trong val/test
        face = faces[len(faces) // 2]
        
        # MTCNN đã normalize về [-1,1] → không cần thêm normalize
        return face, torch.tensor(label, dtype=torch.long)
```

> **Tại sao num_frames=32 thay vì 270?**  
> Kaggle T4 có 16GB VRAM. Với on-the-fly extraction, mỗi sample load 1 frame đại diện.  
> Để tái hiện đúng paper (270 frames/video), cần pre-extract faces ra disk trước (Phase 2 alternative bên dưới).

**Alternative: Pre-extract faces vào /kaggle/working (nếu dataset nhỏ hơn 20GB sau khi extract):**

```python
# Chạy cell này 1 lần để extract toàn bộ faces
# Chỉ extract 32 frames/video → ~1.5M * (32/270) ≈ ~180K ảnh → ~5GB

!python extract_faces_kaggle.py \
    --data_root /kaggle/input/faceforensics-plus-plus-c23 \
    --output_dir /kaggle/working/faces \
    --num_frames 32 \
    --compression c23
```

---

### Phase 4: Training Trên Kaggle (~6-8 giờ)

```python
# Kaggle Notebook — train cell chính

import torch, timm
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda')
MANIPULATION = 'Deepfakes'   # Thay đổi để train các method khác
COMPRESSION  = 'c23'
BATCH_SIZE   = 32
NUM_EPOCHS   = 30
LR           = 2e-4

# Model
model = timm.create_model('xception', pretrained=True)
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))
model = model.to(DEVICE)

# Dataset
train_ds = FaceForensicsDatasetKaggle(
    data_root='/kaggle/input/faceforensics-plus-plus-c23',
    split='train', manipulation=MANIPULATION, compression=COMPRESSION,
    num_frames=32, device=DEVICE
)
val_ds = FaceForensicsDatasetKaggle(
    data_root='/kaggle/input/faceforensics-plus-plus-c23',
    split='val', manipulation=MANIPULATION, compression=COMPRESSION,
    num_frames=32, device=DEVICE
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, 
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5, verbose=True
)

best_val_acc = 0.0
for epoch in range(NUM_EPOCHS):
    # --- Train ---
    model.train()
    train_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    
    # --- Validate ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            val_loss += criterion(out, labels).item()
            val_correct += out.argmax(1).eq(labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = 100. * val_correct / val_total
    scheduler.step(val_loss / len(val_loader))
    
    print(f'Epoch {epoch+1:02d} | '
          f'Train Acc: {100.*correct/total:.1f}% | '
          f'Val Acc: {val_acc:.1f}%')
    
    # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
        }, f'/kaggle/working/checkpoints/best_{MANIPULATION}_{COMPRESSION}.pth')
        print(f'  ✓ Saved checkpoint (val_acc={val_acc:.2f}%)')

print(f'\nBest Val Acc: {best_val_acc:.2f}%')
```

---

### Phase 5: Download Checkpoint Về Local

**Cách 1 — Từ Kaggle UI:**
```
Notebook → Output → Tìm file best_Deepfakes_c23.pth → Download
```

**Cách 2 — Kaggle API (nhanh hơn):**
```bash
# Cài Kaggle API trên máy local
pip install kaggle

# Download output của notebook
kaggle kernels output <your-username>/<notebook-name> -p ./checkpoints/

# File checkpoint sẽ ở: ./checkpoints/best_Deepfakes_c23.pth (~90MB)
```

**Cách 3 — Trong notebook, zip và push lên Google Drive:**
```python
# Cuối notebook, chạy cell này để backup lên Drive
from google.colab import drive  # Không dùng trong Kaggle
# Hoặc dùng kaggle dataset update để lưu lại
import shutil
shutil.copy(
    '/kaggle/working/checkpoints/best_Deepfakes_c23.pth',
    '/kaggle/working/best_Deepfakes_c23.pth'  # Kaggle tự expose file này
)
```

---

### Phase 6: Evaluation Trên Kaggle (Tùy Chọn)

```python
# Chạy evaluation ngay trên Kaggle trước khi download
from sklearn.metrics import accuracy_score, roc_auc_score

model.eval()
all_probs, all_labels = [], []

test_ds = FaceForensicsDatasetKaggle(
    data_root='/kaggle/input/faceforensics-plus-plus-c23',
    split='test', manipulation=MANIPULATION, compression=COMPRESSION,
    num_frames=32, device=DEVICE
)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        probs = torch.softmax(out, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
auc = roc_auc_score(all_labels, all_probs)
print(f'Test Accuracy: {acc*100:.2f}%')
print(f'Test AUC-ROC:  {auc:.4f}')
```

---

## 12. Đánh Giá & Benchmark

### 12.1 Metrics Chính

```python
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np

def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (np.array(y_pred_prob) > threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'auc_roc': roc_auc_score(y_true, y_pred_prob),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    return metrics
```

### 12.2 Kết Quả Kỳ Vọng (Từ Paper)

Accuracy (%) trên **c23** (HQ compression):

| Method | DF | F2F | FS | NT | ALL |
|--------|-----|-----|-----|-----|-----|
| XceptionNet | **99.26** | **99.43** | **99.80** | **98.85** | **99.34** |
| MesoInception4 | 98.40 | 83.58 | 92.83 | 83.05 | 89.47 |
| FWA (FaceWarping) | 97.97 | 52.95 | 67.00 | 54.55 | - |

Accuracy (%) trên **c40** (LQ compression - thử thách nhất):

| Method | DF | F2F | FS | NT | ALL |
|--------|-----|-----|-----|-----|-----|
| XceptionNet | **95.73** | **94.74** | **96.28** | **84.47** | **92.81** |
| MesoInception4 | 84.64 | 71.25 | 83.18 | 68.02 | 76.77 |

> **Quan trọng**: Khi tái hiện, target accuracy phải trong khoảng ±1% của các số trên.

### 12.3 Detection Rate dưới Compression

```python
# Test model trên c0 nhưng áp dụng compression khác nhau khi inference
def test_compression_robustness(model, video_path, compressions=[0, 23, 40]):
    results = {}
    for crf in compressions:
        # Compress video với ffmpeg
        compressed_path = f'/tmp/test_crf{crf}.mp4'
        os.system(f'ffmpeg -i {video_path} -c:v libx264 -crf {crf} {compressed_path}')
        
        score, pred = predict_video(model, compressed_path, ...)
        results[f'crf_{crf}'] = {'score': score, 'prediction': pred}
    
    return results
```

### 12.4 Verification Checklist

```
□ Dataset extraction: ~1.5M face crops extracted thành công
□ Train split: 720 real + 720 fake videos/manipulation
□ Val accuracy (c23, DF): > 98% sau epoch 5
□ Test accuracy (c23, DF): > 99%
□ Test accuracy (c40, DF): > 94%
□ XceptionNet > MesoInception4 trên tất cả settings
□ Performance giảm dần: c0 > c23 > c40
□ Cross-manipulation performance thấp hơn same-manipulation
□ Video-level accuracy tương đương hoặc cao hơn frame-level
```

### 12.5 Comparison: Human vs AI Detector

Paper cũng báo cáo **human performance**:

| Evaluator | DF | F2F | FS | NT |
|-----------|-----|-----|-----|-----|
| Untrained human | ~53% | ~68% | ~53% | ~58% |
| Forensics expert | ~66% | ~82% | ~67% | ~73% |
| XceptionNet (c23) | 99.26% | 99.43% | 99.80% | 98.85% |

Khi tái hiện, human baselines **không cần** tái hiện nhưng context này giúp validate rằng task là có ý nghĩa.

---

## 13. Rủi Ro, Điểm Mơ Hồ & Cách Xử Lý

### 13.1 Điểm Mơ Hồ Trong Paper

| Vấn Đề | Mô Tả | Giải Pháp Đề Xuất |
|---------|-------|-------------------|
| **Frame sampling** | Paper không nêu rõ số frames/video khi training | Dùng **270 frames/video** (uniform sampling). Tham khảo code chính thức của tác giả trên GitHub |
| **Face crop margin** | Paper chỉ nói "extract face regions" | Dùng margin **0.4** (40% mỗi phía). Đây là giá trị phổ biến nhất trong các implementations |
| **Batch size** | Không được đề cập | **32** cho XceptionNet trên 1×GPU 16GB, **16** nếu GPU nhỏ hơn |
| **LR schedule** | Chỉ nói "Adam với lr=0.0002" | ReduceLROnPlateau với patience=5 là lựa chọn an toàn |
| **Training epochs** | Không được đề cập | Early stopping với patience=10 dựa trên val loss |
| **Data augmentation** | Không nêu chi tiết | RandomHorizontalFlip + ColorJitter nhẹ là standard |
| **Input normalization** | Chỉ nói resize về 299×299 | Dùng [-1,1] range (mean=0.5, std=0.5) như XceptionNet gốc |

### 13.2 Vấn Đề Dataset

```
⚠️ Deepfakes crawled từ internet: Một số video có thể bị xóa
→ Giải pháp: Dùng official download script của tác giả, họ maintain dataset

⚠️ Video ID mapping: Fake videos dùng format "XXX_YYY" (source_target)
→ Giải pháp: Parse từ filename, không hardcode

⚠️ Không phải tất cả 1000 video đều có đủ 4 manipulations
→ Giải pháp: Verify sau khi download, skip missing pairs
```

### 13.3 Vấn Đề Kỹ Thuật

```python
# ISSUE: dlib face detector chậm và không phát hiện face nhỏ
# SOLUTION: Dùng RetinaFace hoặc MTCNN thay thế
# pip install retinaface-pytorch
from retinaface import RetinaFace

# ISSUE: Memory overflow khi load toàn bộ dataset vào RAM
# SOLUTION: Dùng DataLoader với num_workers và disk-based dataset

# ISSUE: Class imbalance khi train "all" manipulations
# Real: 720 videos
# Fake: 720*4 = 2880 videos → imbalance 1:4
# SOLUTION: WeightedRandomSampler hoặc oversample real class
from torch.utils.data import WeightedRandomSampler

def get_balanced_sampler(dataset):
    labels = [s[1] for s in dataset.samples]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights, len(weights))

# ISSUE: XceptionNet từ timm có thể khác implementation gốc
# SOLUTION: Verify với tác giả's GitHub hoặc dùng:
# https://github.com/ondyari/FaceForensics
```

### 13.4 Rủi Ro Khi Tái Hiện

| Rủi Ro | Xác Suất | Tác Động | Mitigation |
|--------|----------|----------|------------|
| Không access dataset | Cao (cần approval) | Cao | Điền form sớm (1-2 ngày) |
| Kaggle session hết giờ (12h) giữa chừng | Trung bình | Trung bình | Save checkpoint mỗi epoch, dùng Persistence=On |
| /kaggle/working đầy (20GB limit) | Cao khi extract faces | Cao | Dùng on-the-fly extraction, không lưu faces ra disk |
| Kết quả thấp hơn paper 2-3% | Trung bình | Trung bình | Tăng num_frames, tune hyperparams |
| GPU OOM với batch_size=32 | Thấp (T4 16GB đủ) | Thấp | Giảm xuống batch_size=16 |
| Face detector miss nhiều faces | Trung bình | Cao | MTCNN tốt hơn dlib, đã dùng sẵn |
| Upload 35GB lên Kaggle chậm | Cao | Thấp | Dùng Kaggle API, upload ban đêm |

### 13.5 Sự Khác Biệt Có Thể Chấp Nhận

Sai số so với paper gốc:
- **±2%** accuracy cho c0 và c23: **Chấp nhận được**
- **±3-5%** accuracy cho c40: **Chấp nhận được** (do compression rất nhạy cảm)
- **> 5%** sai lệch: Cần debug lại preprocessing hoặc normalization

---

## 14. Snippets Code Tham Khảo

### 14.1 Quick Start (Toàn Bộ Pipeline)

```python
# quick_train.py — Chạy toàn bộ pipeline với dataset nhỏ để test
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import timm

# 1. Model
model = timm.create_model('xception', pretrained=True, num_classes=2)
# Add dropout before final FC
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))

# 2. Config — paths theo Kaggle workflow
config = {
    # Trên Kaggle:
    'data_root': '/kaggle/input/faceforensics-plus-plus-c23',
    'checkpoint_dir': '/kaggle/working/checkpoints',
    # Trên local (sau khi download checkpoint):
    # 'data_root': './data/FaceForensics++',
    # 'checkpoint_dir': './checkpoints',
    'manipulation': 'Deepfakes',
    'compression': 'c23',
    'batch_size': 32,
    'lr': 2e-4,
    'num_epochs': 30,
    'device': 'cuda'
}

# 3. Dataset + DataLoader
train_ds = FaceForensicsDataset(**config, split='train', transform=train_transforms)
val_ds   = FaceForensicsDataset(**config, split='val',   transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

# 4. Training
device = torch.device('cuda')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

for epoch in range(30):
    # train + val
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    print(f'Epoch {epoch+1}: Train={train_acc:.1f}%, Val={val_acc:.1f}%')
```

### 14.2 Grad-CAM Visualization

```python
# Visualize vùng model chú ý (để debug và paper figures)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

model.eval()
target_layers = [model.conv4]  # Last conv layer của XceptionNet

cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=img_tensor)

visualization = show_cam_on_image(
    img_rgb.astype(np.float32) / 255,
    grayscale_cam[0],
    use_rgb=True
)
```

### 14.3 Cross-Manipulation Evaluation Matrix

```python
import pandas as pd

manipulations = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
results = {}

for train_manip in manipulations:
    results[train_manip] = {}
    model = load_model(f'checkpoints/xception_c23_{train_manip}/best.pth')
    
    for test_manip in manipulations:
        acc, auc = evaluate_on_test_set(
            model, data_root, test_manip, 'c23', device
        )
        results[train_manip][test_manip] = acc

df = pd.DataFrame(results)
print("Cross-manipulation accuracy matrix:")
print(df.to_string())
```

### 14.4 Compression Robustness Test

```python
def apply_h264_compression(input_path, output_path, crf):
    """Compress video với H.264."""
    import subprocess
    cmd = [
        'ffmpeg', '-i', input_path,
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-preset', 'medium',
        '-y', output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

# Test trên 3 levels
for crf in [0, 23, 40]:
    apply_h264_compression('test_video.mp4', f'test_crf{crf}.mp4', crf)
    score, pred = predict_video(model, f'test_crf{crf}.mp4', ...)
    print(f'CRF={crf}: {pred} (score={score:.3f})')
```

### 14.5 Kaggle → Local: Download & Verify Checkpoint

```python
# Verify checkpoint sau khi download về local
import torch
import timm
import torch.nn as nn

checkpoint_path = './checkpoints/best_Deepfakes_c23.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"Epoch saved: {checkpoint['epoch']}")
print(f"Val accuracy: {checkpoint['val_acc']:.2f}%")
print(f"Keys: {list(checkpoint.keys())}")

# Load model
model = timm.create_model('xception', pretrained=False)
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✓ Checkpoint loaded thành công")
```

---

## 15. Local Inference Sau Khi Download Checkpoint

Sau khi download checkpoint về local, **không cần GPU** để chạy inference trên ảnh/video đơn lẻ (CPU đủ dùng cho demo).

### 15.1 Requirements Local (Tối Giản)

```bash
# Chỉ cần các thư viện này để chạy inference — không cần CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm
pip install opencv-python
pip install facenet-pytorch    # MTCNN face detector
pip install numpy
```

### 15.2 Inference Script Local

```python
# local_inference.py
# Chạy trên máy local sau khi download checkpoint từ Kaggle

import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import transforms

# ── CONFIG ──────────────────────────────────────────────
CHECKPOINT_PATH = './checkpoints/best_Deepfakes_c23.pth'
DEVICE = torch.device('cpu')   # CPU đủ dùng cho inference
# ────────────────────────────────────────────────────────

# Load model
def load_model(checkpoint_path, device):
    model = timm.create_model('xception', pretrained=False)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✓ Model loaded (val_acc={ckpt['val_acc']:.2f}%)")
    return model

# Face detector
mtcnn = MTCNN(image_size=299, margin=40, keep_all=False, device=DEVICE)

# Transform (MTCNN đã normalize về [-1,1], nên không cần thêm)
def predict_image(model, image_path, device):
    """Predict 1 ảnh."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    face_tensor = mtcnn(img_rgb)
    if face_tensor is None:
        print("⚠️  Không phát hiện khuôn mặt trong ảnh")
        return None
    
    face_tensor = face_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(face_tensor)
        prob_fake = torch.softmax(output, dim=1)[0, 1].item()
    
    label = 'FAKE' if prob_fake > 0.5 else 'REAL'
    confidence = prob_fake if prob_fake > 0.5 else 1 - prob_fake
    
    print(f"Kết quả: {label} (confidence: {confidence:.1%}, P(fake)={prob_fake:.4f})")
    return {'label': label, 'prob_fake': prob_fake, 'confidence': confidence}


def predict_video(model, video_path, device, num_frames=32, threshold=0.5):
    """Predict video bằng cách average frame predictions."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {total} frames, {fps:.1f} FPS, "
          f"duration={total/fps:.1f}s")
    
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frame_probs = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_tensor = mtcnn(frame_rgb)
        
        if face_tensor is None:
            continue
        
        face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(face_tensor)
            prob_fake = torch.softmax(out, dim=1)[0, 1].item()
        frame_probs.append(prob_fake)
    
    cap.release()
    
    if not frame_probs:
        print("⚠️  Không detect được face trong video")
        return None
    
    avg_prob = np.mean(frame_probs)
    label = 'FAKE' if avg_prob > threshold else 'REAL'
    confidence = avg_prob if avg_prob > threshold else 1 - avg_prob
    face_detected = len(frame_probs)
    
    print(f"Frames analyzed: {face_detected}/{num_frames}")
    print(f"Kết quả: {label} (confidence: {confidence:.1%}, P(fake)={avg_prob:.4f})")
    
    return {
        'label': label,
        'prob_fake': avg_prob,
        'confidence': confidence,
        'frames_analyzed': face_detected,
        'frame_probs': frame_probs
    }


# ── MAIN ────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    
    model = load_model(CHECKPOINT_PATH, DEVICE)
    
    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    if input_path is None:
        print("Usage: python local_inference.py <image_or_video_path>")
        print("Example: python local_inference.py test_video.mp4")
        sys.exit(1)
    
    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        result = predict_video(model, input_path, DEVICE)
    else:
        result = predict_image(model, input_path, DEVICE)
```

### 15.3 Tốc Độ Inference Local (Ước Tính)

| Hardware | Thời gian/frame | Thời gian video 30s |
|----------|----------------|---------------------|
| CPU (modern) | ~200ms | ~1.5 phút (32 frames) |
| GPU RTX 3060 | ~15ms | ~7 giây (32 frames) |
| Kaggle T4 | ~10ms | ~5 giây (32 frames) |

> Với CPU, inference 1 ảnh ~200ms là chấp nhận được cho demo. Nếu cần nhanh hơn: giảm `num_frames=10` hoặc dùng model nhỏ hơn (MesoNet).

### 15.4 Cấu Trúc Thư Mục Local Sau Khi Download

```
faceforensics-local/
├── checkpoints/
│   ├── best_Deepfakes_c23.pth       # ~90MB
│   ├── best_Face2Face_c23.pth       # ~90MB
│   ├── best_FaceSwap_c23.pth        # ~90MB
│   └── best_NeuralTextures_c23.pth  # ~90MB
├── local_inference.py               # Script inference
├── requirements_local.txt           # Dependencies tối giản (CPU)
└── test_samples/                    # Ảnh/video để test
    ├── real_sample.mp4
    └── fake_sample.mp4
```

**requirements_local.txt:**
```
torch>=2.0.0                   # CPU version, nhẹ hơn CUDA version
torchvision>=0.15.0
timm>=0.9.0
opencv-python>=4.8.0
facenet-pytorch>=2.5.3
numpy>=1.24.0
```

---

## Tài Liệu Tham Khảo Kỹ Thuật

- **Repository chính thức**: https://github.com/ondyari/FaceForensics
- **XceptionNet paper**: Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. CVPR.
- **MesoNet paper**: Afchar et al. (2018). MesoNet: a Compact Facial Video Forgery Detection Network.
- **Face2Face paper**: Thies et al. (2016). Face2Face: Real-time face capture and reenactment of RGB videos. CVPR.
- **NeuralTextures paper**: Thies et al. (2019). Deferred Neural Rendering. SIGGRAPH.
- **FaceSwap**: https://github.com/MarekKowalski/FaceSwap
- **dlib**: http://dlib.net
- **timm (PyTorch Image Models)**: https://github.com/huggingface/pytorch-image-models

---

## Checklist Hoàn Thành

```
PHASE 0 - ĐĂNG KÝ DATASET
□ Form xin quyền truy cập đã được submit
□ Email chứa download link đã nhận được
□ Dataset c23 đã download (~35GB)

PHASE 1 - KAGGLE SETUP
□ Dataset đã upload lên Kaggle Dataset
□ Kaggle Notebook tạo với GPU T4
□ Dataset mount tại /kaggle/input/... thành công
□ timm, facenet-pytorch cài được trong notebook

PHASE 2 - TRAINING TRÊN KAGGLE
□ DataLoader load video + extract face on-the-fly OK
□ Model forward pass không error (batch_size=32 vừa 16GB VRAM)
□ Training loop chạy ổn định, không OOM
□ Val accuracy tăng theo epoch
□ Best checkpoint được lưu tại /kaggle/working/checkpoints/

PHASE 3 - DOWNLOAD & VERIFY
□ Checkpoint download về local (~90MB/model)
□ torch.load() thành công, val_acc hiển thị đúng
□ Model forward pass trên CPU không error

PHASE 4 - LOCAL INFERENCE
□ local_inference.py chạy được trên CPU
□ Ảnh real → predict REAL ✓
□ Ảnh fake → predict FAKE ✓
□ Video inference trả về kết quả hợp lý

PHASE 5 - EVALUATION (TRÊN KAGGLE)
□ Test accuracy trong khoảng ±2% so với paper
□ Deepfakes c23 target: >99%
□ Ranking: XceptionNet > MesoNet ✓
□ Cross-manipulation test chạy được
```

---

*Tài liệu này được tổng hợp từ paper FaceForensics++ (Rössler et al., ICCV 2019) và các best practices cộng đồng. Workflow: **Train trên Kaggle GPU T4 → Download checkpoint → Inference local (CPU)**. Phiên bản cuối cùng nên được kiểm tra lại với [code chính thức](https://github.com/ondyari/FaceForensics) của tác giả.*
