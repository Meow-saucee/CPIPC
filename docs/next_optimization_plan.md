# 下一轮优化预案

本文用于当前 `yolo11n_seg_scalecombo_best_candidate` 训练完成后，根据实际审计结果快速决定下一步。不要在最终评估完成前宣称指标达标。

## 1. 判断顺序

训练完成后，先看：

```text
outputs/reports/delivery_audit_yolo11n_seg_scalecombo_best_candidate.md
outputs/reports/candidate_comparison.md
outputs/reports/submission_metrics_yolo11n_seg_scalecombo_best_candidate_val.json
outputs/reports/speed_buckets_yolo11n_seg_scalecombo_best_candidate_test.json
```

判断优先级：

1. `tiny_recall_at_iou50 >= 0.90`
2. `mAP50` 是否高于参考候选
3. `recall_at_iou50` 是否保持稳定
4. 常规图最大耗时是否 `<100ms`
5. 超大图最大耗时是否 `<2000ms`
6. 交付包是否包含 `.pth` 权重、提交 JSON、源码、配置、报告和复现说明
7. `large_mean_matched_iou` 和 `large_mean_best_iou` 作为大裂纹定位诊断指标保留，不再作为 0.85 硬门槛

## 2. 如果 Tiny Recall 不达标

优先动作：

```yaml
infer.conf: 0.005
infer.imgsz: 1536
infer.tile_size: 1024
infer.tile_overlap: 256 或 384
```

训练侧动作：

```text
继续保留 scale-aware 采样
提高 tiny crop repeat
尝试 imgsz=1280, batch=1/2
减少过强缩放增强，避免极小裂纹被进一步压小
```

验证命令重点看：

```text
tiny_recall_at_iou50
predicted_boxes
precision_at_conf
```

注意：降低 `conf` 往往提高召回但增加误检，最终仍要看 `mAP50` 和提交口径。

## 3. 如果大裂纹定位参考指标明显偏低

该项不再是必须优化到 0.85 的硬门槛。只有当大裂纹错误样本明显拖累 mAP50 或答辩需要解释定位质量时，再做以下消融：

```yaml
infer.include_global_for_large: true
infer.global_max_side: 1536 或 1792
infer.adaptive_global_max_side: 1536
infer.adaptive_global_threshold: 4096
infer.box_expand_ratio: 0.08 到 0.16
infer.elongated_box_expand_ratio: 0.1 到 0.25
infer.union_elongated_clusters: true
infer.union_cluster_box_source: mask_box
infer.union_cluster_score_floor: 0.5 到 0.8
```

诊断命令：

```bash
python src/diagnose_large_iou.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --submit outputs/submissions/val_pred_yolo11n_seg_scalecombo_best_candidate.json \
  --split val \
  --out-csv outputs/reports/large_iou_diagnosis_yolo11n_scalecombo.csv \
  --out-json outputs/reports/large_iou_diagnosis_yolo11n_scalecombo.json
```

如果 `large_mean_best_iou` 高但 `large_mean_matched_iou` 低，说明好框存在但排序分数不够，可调 union/variant 分数。

如果 `large_mean_best_iou` 也低，说明候选框形状本身不准，可提高全局分支分辨率、加强大裂纹 crop 训练或换更强模型。

## 4. 如果常规图单张耗时不达标

优先动作：

```yaml
infer.direct_resize_max_side: 960
infer.tile_trigger: low_preds
infer.max_tiles: 4
infer.global_max_side: 1024
```

代码侧动作：

```text
检查 speed_buckets details，定位最慢的 regular 图片。
如果慢图是首次 warmup，可在提交耗时统计中单独记录 warmup 影响，但审计仍按严格规则报告。
如果慢图是接近 2048 的 BMP/TIFF，优先 direct resize 到 960 或 1024。
```

候选对比：

```bash
python src/compare_candidates.py \
  --candidate yolo26n_ref_unionfloor05 \
    outputs/reports/submission_metrics_seg_ref_yolo26n_hybrid_unionfloor05_val.json \
    outputs/reports/inference_time_summary_seg_ref_yolo26n_hybrid_unionfloor05.json \
    outputs/reports/speed_buckets_seg_ref_yolo26n_hybrid_unionfloor05_test.json \
    outputs/reports/delivery_audit_yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate.json \
  --candidate yolo11n_scalecombo \
    outputs/reports/submission_metrics_yolo11n_seg_scalecombo_best_candidate_val.json \
    outputs/reports/inference_time_summary_yolo11n_seg_scalecombo_best_candidate.json \
    outputs/reports/speed_buckets_yolo11n_seg_scalecombo_best_candidate_test.json \
    outputs/reports/delivery_audit_yolo11n_seg_scalecombo_best_candidate.json
```

## 5. 如果整体 mAP50 低于参考候选

优先动作：

```text
不要直接替换最终推荐候选。
保留 YOLO26n 参考候选作为提交基线。
把 yolo11n scalecombo 作为消融实验和工程化补充。
```

可继续尝试：

```text
使用 yolo11s-seg 或 yolo26n-seg 做 scale-aware + scale-crop 长训。
提高训练 imgsz 到 1280。
减少 mixup/erasing，观察细裂纹定位。
用参考 YOLO26n 权重的后处理参数做新模型阈值搜索。
```

## 6. 推荐下一组实验

如果当前 `yolo11n` 不优于参考候选，下一组建议：

```text
实验 A：yolo11s-seg, imgsz=1024, batch=2, scale-aware + scale-crop, epochs=200
实验 B：yolo11n-seg, imgsz=1280, batch=1, scale-aware + scale-crop, epochs=200
实验 C：参考 YOLO26n 权重 + large global 1536/1792 + direct_resize 960 后处理搜索
```

优先级：

```text
C 用时最短，适合先做推理后处理搜索。
A 可能提升整体 mAP 和 mask 质量。
B 可能提升小裂纹召回，但速度和显存压力更大。
```
