#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# A1: YOLO11s-seg @ imgsz=1280, 200 epochs, scaleaware_scalecrop
# ============================================================
# 改动 vs A0: YOLO11n→YOLO11s, imgsz 1024→1280
# 预期收益: mAP50 +0.05~0.10（更大 backbone + 更高分辨率）
# 训练时间: 4080 ~2.5h, 4060Ti ~4-5h
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---- 可覆盖参数 ----
CONFIG="${CONFIG:-configs/yolo_seg_crack_hybrid.yaml}"
DATA_YAML="${DATA_YAML:-data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml}"
MODEL="${MODEL:-yolo11s-seg.pt}"           # Ultralytics 预训练
IMGSZ="${IMGSZ:-1280}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-2}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-42}"
TAG="${TAG:-a1-yolo11s-scaleaware-scalecrop}"
RUN_NAME="${RUN_NAME:-yolo11s-seg_cpipc-chip-crack-seg_img1280_ep200_bs2_seed42_a1}"

echo "========================================"
echo " A1 Training: YOLO11s-seg @ imgsz=1280"
echo "========================================"
echo "Config:      $CONFIG"
echo "Data:        $DATA_YAML"
echo "Model:       $MODEL"
echo "Image size:  $IMGSZ"
echo "Epochs:      $EPOCHS"
echo "Batch:       $BATCH"
echo "Device:      $DEVICE"
echo "Seed:        $SEED"
echo "Run name:    $RUN_NAME"
echo "========================================"

# 确认预训练权重存在
if [ ! -f "$MODEL" ]; then
    echo "Downloading $MODEL ..."
    python -c "from ultralytics import YOLO; YOLO('$MODEL'); print('ok')"
fi

python src/train_yolo_seg.py \
    --config "$CONFIG" \
    --data-yaml "$DATA_YAML" \
    --model "$MODEL" \
    --imgsz "$IMGSZ" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --tag "$TAG" \
    --name "$RUN_NAME"

echo "========================================"
echo " A1 training completed."
echo " Weights at: runs/crack_yolo_seg/$RUN_NAME/weights/best.pt"
echo "========================================"
