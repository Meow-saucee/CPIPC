#!/usr/bin/env bash
set -euo pipefail

cd /home/ruiyi/CPIPC/Dection

CONFIG="${CONFIG:-configs/yolo_seg_crack_hybrid.yaml}"
FAST_CONFIG="${FAST_CONFIG:-configs/yolo_seg_crack_fast.yaml}"
DATASET="${DATASET:-dataset}"
SPLIT="${SPLIT:-test}"

FINAL_OUT="${OUT:-outputs/submissions/results_ensemble_weighted_route_regular_gt100_fastdetbox768_warm_reproduced.json}"
WEIGHTED_OUT="${WEIGHTED_OUT:-outputs/submissions/reproduce_${SPLIT}_ensemble_weighted.json}"
FAST_OUT="${FAST_OUT:-outputs/submissions/reproduce_${SPLIT}_yolo11n_fast_detbox768_warm.json}"

Y26_WEIGHTS="${Y26_WEIGHTS:-deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/weights/yolo26n_ref_unionfloor05.pth}"
Y11_WEIGHTS="${Y11_WEIGHTS:-deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/weights/yolo11n_scalecombo_best.pth}"

SLOW_IDS="${SLOW_IDS:-2,522,526,533,536,548,1038,1063,1072,1079,1088,1236,1304,1506,1513,1725,1733,1762,1898,1915,1922,1977,1980,2074}"

Y26_OUT="${Y26_OUT:-outputs/submissions/reproduce_${SPLIT}_yolo26n_ref_unionfloor05.json}"
Y11_OUT="${Y11_OUT:-outputs/submissions/reproduce_${SPLIT}_yolo11n_scalecombo.json}"

python src/infer_submit_seg.py \
  --config "$CONFIG" \
  --weights "$Y26_WEIGHTS" \
  --split "$SPLIT" \
  --direct-resize-max-side 960 \
  --warmup 5 \
  --out "$Y26_OUT"

python src/infer_submit_seg.py \
  --config "$CONFIG" \
  --weights "$Y11_WEIGHTS" \
  --split "$SPLIT" \
  --direct-resize-max-side 960 \
  --warmup 5 \
  --out "$Y11_OUT"

python src/merge_submissions.py \
  --inputs "$Y26_OUT" "$Y11_OUT" \
  --out "$WEIGHTED_OUT" \
  --iou-thr 0.65 \
  --mode weighted \
  --max-preds 300 \
  --dataset "$DATASET"

python src/infer_submit_seg.py \
  --config "$FAST_CONFIG" \
  --weights "$Y11_WEIGHTS" \
  --split "$SPLIT" \
  --imgsz 768 \
  --conf 0.08 \
  --tile-size 0 \
  --prediction-box-source det_box \
  --no-retina-masks \
  --no-keep-masks-for-merge \
  --warmup 5 \
  --out "$FAST_OUT"

python src/route_by_ids.py \
  --base "$WEIGHTED_OUT" \
  --alternate "$FAST_OUT" \
  --ids "$SLOW_IDS" \
  --out "$FINAL_OUT"

python src/check_submit.py \
  --dataset "$DATASET" \
  --submit "$FINAL_OUT"

python src/summarize_submission_time.py \
  --submit "$FINAL_OUT" \
  --out-json "outputs/reports/inference_time_summary_reproduced_final_speed_route.json" \
  --out-csv "outputs/reports/inference_time_per_image_reproduced_final_speed_route.csv"

python src/analyze_submission_speed.py \
  --dataset "$DATASET" \
  --submit "$FINAL_OUT" \
  --split "$SPLIT" \
  --out "outputs/reports/speed_buckets_reproduced_final_speed_route_${SPLIT}.json"

echo "Reproduced final speed-route submission: $FINAL_OUT"
