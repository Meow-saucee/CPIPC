#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# A1 完整管线：训练 → 推理 → Ensemble → Speed Route → 审计
# ============================================================
# vs A0 差异:
#   - YOLO11s 替换 YOLO11n
#   - imgsz=1280 替换 imgsz=1024
#   - slow regular ID 动态识别，不复用 A0 硬编码列表
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---- 路径 ----
A0_DELIVERY="deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate"
Y26_WEIGHTS="$A0_DELIVERY/weights/yolo26n_ref_unionfloor05.pth"
A1_RUN_NAME="${A1_RUN_NAME:-yolo11s-seg_cpipc-chip-crack-seg_img1280_ep200_bs2_seed42_a1}"
A1_WEIGHTS="${A1_WEIGHTS:-runs/crack_yolo_seg/$A1_RUN_NAME/weights/best.pt}"
A1_FAST_WEIGHTS="${A1_FAST_WEIGHTS:-$A1_WEIGHTS}"

CONFIG="${CONFIG:-configs/yolo_seg_crack_hybrid.yaml}"
FAST_CONFIG="${FAST_CONFIG:-configs/yolo_seg_crack_fast.yaml}"
DATASET="${DATASET:-dataset}"
SPLIT="${SPLIT:-test}"
OUT_DIR="${OUT_DIR:-outputs/a1}"
OUT_VAL_DIR="${OUT_VAL_DIR:-${OUT_DIR}/val}"
mkdir -p "$OUT_DIR" "$OUT_VAL_DIR"

# ---- 输出文件 ----
Y26_TEST="$OUT_DIR/a1_yolo26n_test.json"
Y11S_TEST="$OUT_DIR/a1_yolo11s_test.json"
WEIGHTED_TEST="$OUT_DIR/a1_ensemble_weighted_test.json"
FAST_TEST="$OUT_DIR/a1_fast_detbox768_test.json"
FINAL_TEST="$OUT_DIR/a1_final_speed_route_test.json"
SPEED_BENCH="$OUT_DIR/a1_speed_benchmark_weighted.json"
SLOW_IDS_FILE="$OUT_DIR/a1_slow_regular_ids.txt"
VAL_PRED="$OUT_VAL_DIR/a1_ensemble_weighted_val.json"
VAL_METRICS="$OUT_DIR/a1_val_metrics.json"
TIME_SUMMARY="$OUT_DIR/a1_inference_time_summary.json"
SPEED_BUCKETS="$OUT_DIR/a1_speed_buckets.json"

echo "========================================"
echo " A1 Full Pipeline"
echo " YOLO26n (fixed) + YOLO11s (new) Ensemble"
echo "========================================"

# ---- Step 0: 训练（如权重不存在） ----
if [ ! -f "$A1_WEIGHTS" ]; then
    echo ""
    echo "=== Step 0: Training YOLO11s ==="
    bash scripts/train_a1.sh
else
    echo "=== Step 0: Weights exist, skip training ==="
    ls -lh "$A1_WEIGHTS"
fi

# ---- Step 1: 验证集推理 + 评估 ----
echo ""
echo "=== Step 1: Validating on val set ==="

Y26_VAL="$OUT_VAL_DIR/a1_yolo26n_val.json"
Y11S_VAL="$OUT_VAL_DIR/a1_yolo11s_val.json"

python src/infer_submit_seg.py \
    --config "$CONFIG" \
    --weights "$Y26_WEIGHTS" \
    --split val \
    --direct-resize-max-side 960 \
    --out "$Y26_VAL"

python src/infer_submit_seg.py \
    --config "$CONFIG" \
    --weights "$A1_WEIGHTS" \
    --split val \
    --direct-resize-max-side 960 \
    --out "$Y11S_VAL"

python src/merge_submissions.py \
    --inputs "$Y26_VAL" "$Y11S_VAL" \
    --out "$VAL_PRED" \
    --iou-thr 0.65 \
    --mode weighted \
    --max-preds 300 \
    --dataset "$DATASET"

python src/eval_submission.py \
    --config "$CONFIG" \
    --submit "$VAL_PRED" \
    --split val \
    --out "$VAL_METRICS"

echo "A1 Val Metrics:"
python -c "
import json
m = json.load(open('$VAL_METRICS'))
for k in ['mAP50','recall_at_iou50','tiny_recall_at_iou50','large_mean_matched_iou','large_mean_best_iou','predicted_boxes']:
    print(f'  {k}: {m.get(k)}')
"

# ---- Step 2: 测试集推理（双模型） ----
echo ""
echo "=== Step 2: Test set inference ==="

python src/infer_submit_seg.py \
    --config "$CONFIG" \
    --weights "$Y26_WEIGHTS" \
    --split "$SPLIT" \
    --direct-resize-max-side 960 \
    --warmup 5 \
    --out "$Y26_TEST"

python src/infer_submit_seg.py \
    --config "$CONFIG" \
    --weights "$A1_WEIGHTS" \
    --split "$SPLIT" \
    --direct-resize-max-side 960 \
    --warmup 5 \
    --out "$Y11S_TEST"

# ---- Step 3: Weighted Merge ----
echo ""
echo "=== Step 3: Weighted merge ==="

python src/merge_submissions.py \
    --inputs "$Y26_TEST" "$Y11S_TEST" \
    --out "$WEIGHTED_TEST" \
    --iou-thr 0.65 \
    --mode weighted \
    --max-preds 300 \
    --dataset "$DATASET"

# ---- Step 4: 测速 + 识别慢 regular 图 ----
echo ""
echo "=== Step 4: Speed benchmark on weighted ensemble ==="

python src/analyze_submission_speed.py \
    --dataset "$DATASET" \
    --submit "$WEIGHTED_TEST" \
    --split "$SPLIT" \
    --out "$SPEED_BENCH"

# 提取 regular 图中单张 > 100ms 的 ID
python -c "
import json
data = json.load(open('$SPEED_BENCH'))
slow = [d['ID'] for d in data.get('details', [])
        if d.get('bucket') == 'regular' and d.get('inference_time_ms', 0) > 100]
with open('$SLOW_IDS_FILE', 'w') as f:
    f.write(','.join(str(x) for x in slow))
print(f'Slow regular images (>100ms): {len(slow)}')
if slow:
    print(f'IDs: {slow}')
"

SLOW_IDS=$(cat "$SLOW_IDS_FILE")
if [ -z "$SLOW_IDS" ]; then
    echo "No slow regular images found, using weighted ensemble directly."
    cp "$WEIGHTED_TEST" "$FINAL_TEST"
else
    # ---- Step 5: Fast branch for slow regular images ----
    echo ""
    echo "=== Step 5: Fast branch for $([ $(echo "$SLOW_IDS" | tr ',' '\n' | wc -l) -gt 0 ] && echo "$SLOW_IDS" | tr ',' '\n' | wc -l || echo 0) slow images ==="

    python src/infer_submit_seg.py \
        --config "$FAST_CONFIG" \
        --weights "$A1_FAST_WEIGHTS" \
        --split "$SPLIT" \
        --imgsz 768 \
        --conf 0.08 \
        --tile-size 0 \
        --prediction-box-source det_box \
        --no-retina-masks \
        --no-keep-masks-for-merge \
        --warmup 5 \
        --out "$FAST_TEST"

    # ---- Step 6: Speed Route ----
    echo ""
    echo "=== Step 6: Speed route ==="

    python src/route_by_ids.py \
        --base "$WEIGHTED_TEST" \
        --alternate "$FAST_TEST" \
        --ids "$SLOW_IDS" \
        --out "$FINAL_TEST"
fi

# ---- Step 7: 提交检查 + 耗时汇总 ----
echo ""
echo "=== Step 7: Submission check & timing ==="

python src/check_submit.py \
    --dataset "$DATASET" \
    --submit "$FINAL_TEST"

python src/summarize_submission_time.py \
    --submit "$FINAL_TEST" \
    --out-json "$TIME_SUMMARY"

python src/analyze_submission_speed.py \
    --dataset "$DATASET" \
    --submit "$FINAL_TEST" \
    --split "$SPLIT" \
    --out "$SPEED_BUCKETS"

# ---- 最终摘要 ----
echo ""
echo "========================================"
echo " A1 Pipeline Complete"
echo "========================================"
echo "Val metrics:  $VAL_METRICS"
echo "Final submit: $FINAL_TEST"
echo "Time summary: $TIME_SUMMARY"
echo "Speed buckets: $SPEED_BUCKETS"
echo ""
echo "=== A1 Val Metrics ==="
python -c "
import json
m = json.load(open('$VAL_METRICS'))
print(f'  mAP50:              {m.get(\"mAP50\"):.4f}')
print(f'  Recall@IoU50:       {m.get(\"recall_at_iou50\"):.4f}')
print(f'  Tiny Recall:        {m.get(\"tiny_recall_at_iou50\"):.4f}')
print(f'  Large Best IoU:     {m.get(\"large_mean_best_iou\"):.4f}')
print(f'  Large Matched IoU:  {m.get(\"large_mean_matched_iou\"):.4f}')
print(f'  Predicted Boxes:    {m.get(\"predicted_boxes\")}')
"
echo ""
echo "=== A1 Time Summary ==="
python -c "
import json
t = json.load(open('$TIME_SUMMARY'))
print(f'  Avg: {t.get(\"avg_inference_time_ms\"):.2f}ms')
print(f'  Max: {t.get(\"max_inference_time_ms\"):.2f}ms')
"
echo "========================================"
