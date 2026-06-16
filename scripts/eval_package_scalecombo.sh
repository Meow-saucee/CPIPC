#!/usr/bin/env bash
set -euo pipefail

cd /home/ruiyi/CPIPC/Dection

RUN_DIR="${1:-runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop}"
NAME="${2:-yolo11n_seg_scalecombo_best_candidate}"
CONFIG="${CONFIG:-configs/yolo_seg_crack_hybrid.yaml}"
WEIGHTS="$RUN_DIR/weights/best.pt"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Missing weights: $WEIGHTS" >&2
  exit 1
fi

VAL_SUBMIT="outputs/submissions/val_pred_${NAME}.json"
TEST_SUBMIT="outputs/submissions/results_${NAME}.json"
VAL_METRICS="outputs/reports/submission_metrics_${NAME}_val.json"
VAL_ERRORS="outputs/reports/submission_errors_${NAME}_val.csv"
TIME_SUMMARY="outputs/reports/inference_time_summary_${NAME}.json"
TIME_CSV="outputs/reports/inference_time_per_image_${NAME}.csv"
SPEED_BUCKETS="outputs/reports/speed_buckets_${NAME}_test.json"

python src/infer_submit_seg.py \
  --config "$CONFIG" \
  --weights "$WEIGHTS" \
  --split val \
  --direct-resize-max-side 960 \
  --out "$VAL_SUBMIT"

python src/eval_submission.py \
  --config "$CONFIG" \
  --submit "$VAL_SUBMIT" \
  --split val \
  --out "$VAL_METRICS" \
  --errors "$VAL_ERRORS"

python src/infer_submit_seg.py \
  --config "$CONFIG" \
  --weights "$WEIGHTS" \
  --split test \
  --direct-resize-max-side 960 \
  --out "$TEST_SUBMIT"

python src/check_submit.py \
  --dataset dataset \
  --submit "$TEST_SUBMIT"

python src/summarize_submission_time.py \
  --submit "$TEST_SUBMIT" \
  --out-json "$TIME_SUMMARY" \
  --out-csv "$TIME_CSV"

python src/analyze_submission_speed.py \
  --dataset dataset \
  --submit "$TEST_SUBMIT" \
  --split test \
  --out "$SPEED_BUCKETS"

python src/package_delivery.py \
  --name "$NAME" \
  --weights "$WEIGHTS" \
  --submission "$TEST_SUBMIT" \
  --config "$CONFIG" \
  --metrics "$VAL_METRICS" \
  --errors "$VAL_ERRORS" \
  --time-summary "$TIME_SUMMARY" \
  --speed-buckets "$SPEED_BUCKETS" \
  --report docs/technical_design_report.md \
  --copy-docs \
  --copy-source

python src/audit_delivery.py \
  --delivery "deliverables/$NAME" \
  --dataset dataset \
  --out-json "outputs/reports/delivery_audit_${NAME}.json" \
  --out-md "outputs/reports/delivery_audit_${NAME}.md"
