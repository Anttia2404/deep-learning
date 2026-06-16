#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-./data/FaceForensics++}

for manipulation in Deepfakes Face2Face FaceSwap NeuralTextures all; do
  python -m evaluation.evaluate \
    --checkpoint "checkpoints/best_${manipulation}_c23.pth" \
    --data-root "$DATA_ROOT" \
    --model xception \
    --manipulation "$manipulation" \
    --compression c23
done
