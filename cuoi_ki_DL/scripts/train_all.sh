#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-./data/FaceForensics++}

python -m training.trainer --config configs/xception_c23.yaml --data-root "$DATA_ROOT" --manipulation Deepfakes
python -m training.trainer --config configs/xception_c23.yaml --data-root "$DATA_ROOT" --manipulation Face2Face
python -m training.trainer --config configs/xception_c23.yaml --data-root "$DATA_ROOT" --manipulation FaceSwap
python -m training.trainer --config configs/xception_c23.yaml --data-root "$DATA_ROOT" --manipulation NeuralTextures
python -m training.trainer --config configs/xception_c23.yaml --data-root "$DATA_ROOT" --manipulation all
