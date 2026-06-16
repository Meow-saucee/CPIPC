#!/usr/bin/env bash
set -euo pipefail

cd /home/ruiyi/CPIPC/Dection

python src/check_yolo_seg_data.py \
  --data-yaml data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml \
  --out outputs/reports/check_crack_seg_scaleaware_scalecrop_before_train.json

python src/train_yolo_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --data-yaml data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml \
  --model /home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/yolo11n-seg.pt \
  --imgsz 1024 \
  --epochs 200 \
  --batch 2 \
  --device 0 \
  --tag seg-scaleaware-scalecrop
