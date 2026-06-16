#!/usr/bin/env bash
set -euo pipefail

# Reproduce the latest seed42_temp0.5 experiment with only:
# - epochs: 1
# - batch_size: 1028
python pretrain_simclr_stl10.py \
  --data-root ./data \
  --save-dir ./outputs/simclr_stl10_seed42_temp05_bs1028_e1 \
  --download \
  --seed 42 \
  --gpu-id 0 \
  --epochs 1 \
  --batch-size 1028 \
  --num-workers 8 \
  --image-size 96 \
  --temperature 0.5 \
  --projection-hidden-dim 2048 \
  --projection-out-dim 128 \
  --lr 0.3 \
  --momentum 0.9 \
  --weight-decay 0.0001 \
  --warmup-epochs 10 \
  --amp \
  --tensorboard \
  --eval-interval 25 \
  --eval-batch-size 256 \
  --linear-eval-epochs 10 \
  --linear-eval-lr 0.1 \
  --knn-k 20 \
  --knn-temperature 0.1 \
  --periodic-keep-last 3 \
  --log-interval 20 \
  --save-every 25
