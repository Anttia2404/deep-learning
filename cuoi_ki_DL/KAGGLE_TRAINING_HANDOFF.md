# Kaggle Training Handoff for FaceForensics++

## Purpose
This document is written so another AI assistant or collaborator can immediately understand the current project state and continue the workflow to train the model on Kaggle.

The user has **already downloaded the full FaceForensics++ c23 video dataset (about 5000 videos total)** and wants to train the project on **Kaggle GPU**.

## Current project context
- Project root: `c:/VoTanTai/cuoi_ki_DL`
- This repository is a Python/PyTorch FaceForensics++ reproduction toolkit.
- Main training entry point: `training/trainer.py`
- Main default config: `configs/xception_c23.yaml`
- Dataset loaders:
  - extracted face crops: `datasets/ff_dataset.py` -> `FaceForensicsDataset`
  - on-the-fly video face extraction: `datasets/ff_dataset.py` -> `FaceForensicsVideoDataset`
- Face detector backend wrapper: `inference/face_detector.py`
- Local inference entry point after training: `local_inference.py`

## Important implementation facts from this repo
1. `training/trainer.py` supports CLI arguments:
   - `--config`
   - `--data-root`
   - `--manipulation`
   - `--compression`
   - `--device`
2. For Kaggle, the safest starting workflow is to use:
   - `use_video_dataset: true`
   - `compression: c23`
   - `manipulation: Deepfakes`
3. When `use_video_dataset: true`, the training script uses `FaceForensicsVideoDataset` and reads videos directly from disk, then extracts representative face crops on-the-fly.
4. The current repo expects split files at:
   - `<DATA_ROOT>/splits/train.json`
   - `<DATA_ROOT>/splits/val.json`
   - `<DATA_ROOT>/splits/test.json`
5. Checkpoints are saved by `training/trainer.py` as:
   - `latest_<manipulation>_<compression>.pth`
   - `best_<manipulation>_<compression>.pth`

## Assumed dataset status
The user says the full dataset has already been downloaded.

Assume the Kaggle dataset should contain at least this structure:

```text
faceforensics-plus-plus-c23/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           └── videos/
├── manipulated_sequences/
│   ├── Deepfakes/
│   │   └── c23/
│   │       └── videos/
│   ├── Face2Face/
│   │   └── c23/
│   │       └── videos/
│   ├── FaceSwap/
│   │   └── c23/
│   │       └── videos/
│   └── NeuralTextures/
│       └── c23/
│           └── videos/
└── splits/
    ├── train.json
    ├── val.json
    └── test.json
```

## Official split references
If the split files are missing, use the official FaceForensics split files:
- `train.json`: https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json
- `val.json`: https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/val.json
- `test.json`: https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json

## Goal for first Kaggle run
Do **not** start with all manipulations at once.

For the first stable run on Kaggle:
- model: `xception`
- manipulation: `Deepfakes`
- compression: `c23`
- use on-the-fly video dataset
- run a manageable number of epochs first

This gives the fastest path to a real checkpoint for demo and local inference.

## Recommended Kaggle workflow

### Step 1 — Upload datasets to Kaggle
Create two **private** Kaggle datasets:

1. **Dataset A: FF++ data**
   - contains the full or partial FaceForensics++ directory
   - suggested name: `ffpp-c23-full`

2. **Dataset B: training code**
   - upload this repository codebase
   - suggested name: `ffpp-training-code`

### Step 2 — Create a Kaggle notebook
Notebook settings:
- Accelerator: `GPU`
- Internet: `On`

Then add both datasets as inputs.

### Step 3 — Install dependencies in Kaggle
Run:

```python
!pip install -q timm facenet-pytorch scikit-learn PyYAML tqdm pandas matplotlib Pillow
```

### Step 4 — Copy code to writable directory
Kaggle inputs are read-only. Copy the code dataset into `/kaggle/working`:

```python
!cp -r /kaggle/input/ffpp-training-code /kaggle/working/ffpp
%cd /kaggle/working/ffpp
```

If the Kaggle dataset slug differs, replace `/kaggle/input/ffpp-training-code` with the actual path.

### Step 5 — Verify dataset path
Check actual Kaggle input names:

```python
!ls /kaggle/input
```

Then verify the FF++ dataset contents:

```python
!ls /kaggle/input/ffpp-c23-full
!ls /kaggle/input/ffpp-c23-full/splits
!ls /kaggle/input/ffpp-c23-full/original_sequences/youtube/c23/videos | head
!ls /kaggle/input/ffpp-c23-full/manipulated_sequences/Deepfakes/c23/videos | head
```

Replace `ffpp-c23-full` if the actual Kaggle slug is different.

## Recommended first Kaggle config
Create a Kaggle-specific config file instead of using `configs/xception_c23.yaml` directly.

Run this cell:

```python
from pathlib import Path

cfg = Path('/kaggle/working/ffpp/configs/kaggle_xception_c23_video.yaml')
cfg.write_text(
"""
model_name: xception
pretrained: true
num_classes: 2
dropout: 0.5
input_size: 299

data_root: /kaggle/input/ffpp-c23-full
manipulation: Deepfakes
compression: c23
batch_size: 16
num_epochs: 10
num_frames_train: 32
num_frames_eval: 32
lr: 0.0002
weight_decay: 0.0
scheduler: reduce_on_plateau
loss_name: cross_entropy
label_smoothing: 0.0
num_workers: 2
checkpoint_dir: /kaggle/working/checkpoints
early_stopping_patience: 5
seed: 42
use_video_dataset: true
detector_backend: mtcnn
device: cuda
""".strip(),
encoding='utf-8'
)
print(cfg.read_text())
```

## Why this config is recommended
- `use_video_dataset: true` because the uploaded data consists of videos, not pre-extracted face crops.
- `num_frames_train: 32` and `num_frames_eval: 32` are lighter and more practical for Kaggle than 270.
- `batch_size: 16` is a safe starting point; reduce to `8` if there is GPU OOM.
- `num_epochs: 10` is good for a first working checkpoint.

## First training command
Run this in Kaggle:

```python
!python -m training.trainer \
  --config /kaggle/working/ffpp/configs/kaggle_xception_c23_video.yaml \
  --data-root /kaggle/input/ffpp-c23-full \
  --manipulation Deepfakes \
  --compression c23 \
  --device cuda
```

Again, replace `/kaggle/input/ffpp-c23-full` if the actual dataset slug is different.

## Expected outputs
The training script should save checkpoints under:

```text
/kaggle/working/checkpoints/
```

Expected files:
- `latest_Deepfakes_c23.pth`
- `best_Deepfakes_c23.pth`

To inspect them:

```python
!ls /kaggle/working/checkpoints
```

To make the best checkpoint easier to download:

```python
import shutil
shutil.copy(
    '/kaggle/working/checkpoints/best_Deepfakes_c23.pth',
    '/kaggle/working/best_Deepfakes_c23.pth',
)
print('Saved to /kaggle/working/best_Deepfakes_c23.pth')
```

## After training
Download `best_Deepfakes_c23.pth` and place it locally in:

```text
c:/VoTanTai/cuoi_ki_DL/checkpoints/
```

Then run local inference, for example:

```bash
python local_inference.py checkpoints/best_Deepfakes_c23.pth path/to/sample.mp4
```

## Troubleshooting

### 1. Kaggle cannot find the dataset path
Run:

```python
!ls /kaggle/input
```

Then update the paths in the config and training command.

### 2. Missing split files
Ensure these exist inside the uploaded dataset:
- `splits/train.json`
- `splits/val.json`
- `splits/test.json`

### 3. GPU out of memory
Reduce:
- `batch_size: 16` -> `8`
- if still needed: `8` -> `4`
- optionally reduce `num_frames_train: 32` -> `16`

### 4. MTCNN dependency issues
If `facenet-pytorch` causes issues in Kaggle, switch to the fallback backend:

```yaml
detector_backend: haar
```

This is usually less accurate than MTCNN, but is more lightweight.

### 5. Training is slow
That is expected with on-the-fly face detection from videos. This is the practical trade-off for avoiding a large face extraction preprocessing stage before Kaggle training.

## Recommended progression after the first successful run
1. Train `Deepfakes + c23` first.
2. Verify checkpoint quality with local inference.
3. Then optionally train:
   - `Face2Face`
   - `FaceSwap`
   - `NeuralTextures`
4. Only after stable single-manipulation runs, consider `manipulation: all`.

## Direct instruction for another AI assistant
If you are another AI reading this file, your job is to:
1. help the user complete Kaggle setup with the uploaded dataset and code,
2. verify the actual Kaggle input paths,
3. create the Kaggle-specific config file,
4. run the first training command for `Deepfakes + c23`,
5. help the user retrieve `best_Deepfakes_c23.pth`,
6. then guide local inference using the saved checkpoint.

Do not start with `manipulation: all` unless the user explicitly asks for it.
Start with the most reliable path: `xception + Deepfakes + c23 + use_video_dataset: true`.
