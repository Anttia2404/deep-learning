# FaceForensics++ Reproduction Toolkit

Codebase Python/PyTorch để tái hiện pipeline phát hiện deepfake theo tài liệu [FaceForensics++_Implementation_Guide.md](FaceForensics++_Implementation_Guide.md).

> Trạng thái hiện tại: repo đã được scaffold đầy đủ để hỗ trợ preprocessing, training, evaluation, và local inference. Các bước phụ thuộc tài nguyên ngoài như xin quyền dataset FaceForensics++, upload Kaggle, train checkpoint thật và benchmark paper vẫn cần được chạy sau khi có dữ liệu.

## Tính năng chính

- XceptionNet fine-tuning cho bài toán real vs fake.
- MesoNet và MesoInception4 làm baseline.
- Dataset loader cho face crops đã extract sẵn.
- Dataset loader tối ưu Kaggle với face extraction on-the-fly từ video.
- Scripts preprocessing để sample frame, detect/crop face, sinh metadata.
- Training loop với checkpointing, early stopping, scheduler.
- Evaluation frame-level và video-level.
- Local inference CPU cho ảnh/video sau khi có checkpoint.

## Cấu trúc thư mục

```text
faceforensics-plus-plus/
├── README.md
├── requirements.txt
├── requirements_local.txt
├── setup.py
├── local_inference.py
├── data/
├── models/
├── datasets/
├── training/
├── evaluation/
├── inference/
├── configs/
├── scripts/
├── tests/
├── notebooks/
└── checkpoints/
```

## Cài đặt môi trường phát triển

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

## Cài đặt tối giản để chỉ chạy inference local

```bash
pip install -r requirements_local.txt
```

## Chuẩn bị dataset FaceForensics++

1. Xin quyền dataset qua form chính thức của FaceForensics++.
2. Dùng script chính thức để tải dữ liệu, ưu tiên `c23` trước.
3. Tổ chức dữ liệu theo cấu trúc:

```text
<DATA_ROOT>/
├── original_sequences/youtube/c23/videos/
├── manipulated_sequences/Deepfakes/c23/videos/
├── manipulated_sequences/Face2Face/c23/videos/
├── manipulated_sequences/FaceSwap/c23/videos/
├── manipulated_sequences/NeuralTextures/c23/videos/
└── splits/
    ├── train.json
    ├── val.json
    └── test.json
```

## Preprocessing

### 1. Extract face crops từ video

```bash
python -m data.extract_faces \
  --data-root /path/to/FaceForensics++ \
  --output-root /path/to/FaceForensics++ \
  --compression c23 \
  --backend mtcnn \
  --num-frames 270
```

### 2. Chạy pipeline preprocess đầy đủ

```bash
python -m data.preprocess_videos \
  --data-root /path/to/FaceForensics++ \
  --output-root ./artifacts/preprocessed \
  --compression c23
```

## Training

Train Xception trên Deepfakes/c23:

```bash
python -m training.trainer --config configs/xception_c23.yaml \
  --data-root /path/to/FaceForensics++ \
  --manipulation Deepfakes
```

Train baseline MesoNet:

```bash
python -m training.trainer --config configs/meso_c0.yaml \
  --data-root /path/to/FaceForensics++ \
  --compression c0
```

## Evaluation

```bash
python -m evaluation.evaluate \
  --checkpoint checkpoints/best_Deepfakes_c23.pth \
  --data-root /path/to/FaceForensics++ \
  --model xception \
  --manipulation Deepfakes \
  --compression c23
```

## Local inference

```bash
python local_inference.py checkpoints/best_Deepfakes_c23.pth /path/to/sample.mp4
```

Hoặc dùng module trực tiếp:

```bash
python -m inference.predict_image \
  --checkpoint checkpoints/best_Deepfakes_c23.pth \
  --input /path/to/image.jpg
```

```bash
python -m inference.predict_video \
  --checkpoint checkpoints/best_Deepfakes_c23.pth \
  --input /path/to/video.mp4
```

## Workflow khuyến nghị Kaggle -> local

1. Tải dataset `c23`.
2. Upload dataset lên Kaggle.
3. Chạy training Xception bằng GPU T4 theo config tương ứng.
4. Download checkpoint tốt nhất về local.
5. Chạy inference CPU bằng `local_inference.py`.

## Giới hạn hiện tại

Các phần dưới đây **chưa thể được xác minh end-to-end trong repo trống này** nếu chưa có dataset/checkpoint thật:

- Download dataset FF++ từ nguồn chính thức.
- Upload Kaggle + training 6-8 giờ.
- So khớp benchmark paper trên test set thật.

Codebase này đã được thiết kế để bạn có thể nối ngay các bước trên khi có quyền truy cập dữ liệu.

## Kiểm thử

```bash
pytest
```

Nếu chưa cài đầy đủ dependency nặng, có thể smoke test cú pháp:

```bash
python -m compileall .
```
