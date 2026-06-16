#!/usr/bin/env bash
set -euo pipefail

cd /home/ruiyi/CPIPC/Dection

CONFIG="${CONFIG:-configs/yolo_seg_crack_hybrid.yaml}"
DATASET="${DATASET:-dataset}"
SPLIT="${SPLIT:-test}"
OUT="${OUT:-outputs/submissions/results_ensemble_w075_calibrated_reproduced.json}"

DELIVERY="${DELIVERY:-deliverables/ensemble_y26_y11_w075_calibrated_candidate}"
Y26_WEIGHTS="${Y26_WEIGHTS:-$DELIVERY/weights/yolo26n_ref_unionfloor05.pth}"
Y11_WEIGHTS="${Y11_WEIGHTS:-$DELIVERY/weights/yolo11n_scalecombo_best.pth}"

Y26_OUT="${Y26_OUT:-outputs/submissions/reproduce_${SPLIT}_yolo26n_ref_unionfloor05.json}"
Y11_OUT="${Y11_OUT:-outputs/submissions/reproduce_${SPLIT}_yolo11n_scalecombo.json}"
RAW_OUT="${RAW_OUT:-outputs/submissions/reproduce_${SPLIT}_ensemble_w075_raw.json}"
CALIBRATED_OUT="${CALIBRATED_OUT:-outputs/submissions/reproduce_${SPLIT}_ensemble_w075_calibrated.json}"

python src/infer_submit_seg.py \
  --config "$CONFIG" \
  --weights "$Y26_WEIGHTS" \
  --split "$SPLIT" \
  --direct-resize-max-side 960 \
  --out "$Y26_OUT"

python src/infer_submit_seg.py \
  --config "$CONFIG" \
  --weights "$Y11_WEIGHTS" \
  --split "$SPLIT" \
  --direct-resize-max-side 960 \
  --out "$Y11_OUT"

python src/merge_submissions.py \
  --inputs "$Y26_OUT" "$Y11_OUT" \
  --out "$RAW_OUT" \
  --iou-thr 0.75 \
  --mode weighted \
  --max-preds 300 \
  --dataset "$DATASET"

python src/calibrate_boxes.py \
  --submit "$RAW_OUT" \
  --out "$CALIBRATED_OUT" \
  --dataset "$DATASET" \
  --min-area 90000 \
  --min-side 700 \
  --min-aspect 3 \
  --scale-x 1.08 \
  --scale-y 1.04 \
  --long-scale 0.96 \
  --short-scale 0.95

python src/demote_oversized_boxes.py \
  --submit "$CALIBRATED_OUT" \
  --out "$OUT" \
  --min-area 90000 \
  --contain-iou 0.55 \
  --area-ratio 1.6 \
  --demote-factor 0.25 \
  --demote-below 0.49 \
  --competitor-score-min 0.45

python src/check_submit.py \
  --dataset "$DATASET" \
  --submit "$OUT"

echo "Reproduced w075 calibrated ensemble submission: $OUT"
