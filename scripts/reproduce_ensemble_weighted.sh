#!/usr/bin/env bash
set -euo pipefail

cd /home/ruiyi/CPIPC/Dection

CONFIG="${CONFIG:-configs/yolo_seg_crack_hybrid.yaml}"
DATASET="${DATASET:-dataset}"
SPLIT="${SPLIT:-test}"
OUT="${OUT:-outputs/submissions/results_ensemble_reproduced.json}"

Y26_WEIGHTS="${Y26_WEIGHTS:-deliverables/ensemble_y26_y11_weighted_candidate/weights/yolo26n_ref_unionfloor05.pth}"
Y11_WEIGHTS="${Y11_WEIGHTS:-deliverables/ensemble_y26_y11_weighted_candidate/weights/yolo11n_scalecombo_best.pth}"

Y26_OUT="${Y26_OUT:-outputs/submissions/reproduce_${SPLIT}_yolo26n_ref_unionfloor05.json}"
Y11_OUT="${Y11_OUT:-outputs/submissions/reproduce_${SPLIT}_yolo11n_scalecombo.json}"

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
  --out "$OUT" \
  --iou-thr 0.65 \
  --mode weighted \
  --max-preds 300 \
  --dataset "$DATASET"

python src/check_submit.py \
  --dataset "$DATASET" \
  --submit "$OUT"

echo "Reproduced ensemble submission: $OUT"
