# 实验结果汇总

本文记录当前工程中已经实际跑通的候选模型、验证指标、提交文件和耗时结果。所有指标均来自本机当前工程脚本，不编造未运行结果。

说明：本文保留了实验过程中的历史记录，其中部分早期实验曾以 Large IoU 0.85 作为优化目标。按当前项目口径，最终候选选择改为 mAP50、Recall、Tiny Recall 和速度优先，Large IoU 仅作为大裂纹定位质量诊断指标。

## 1. 候选模型

### YOLO11n-seg 20 epoch 基准

- 权重：`runs/crack_yolo_seg/yolo11n_seg_img1024_ep20_bs2_baseline/weights/best.pt`
- 训练配置：`imgsz=1024, epochs=20, batch=2`
- 交付包：`deliverables/yolo11n_seg_ep20_fast_scaled`
- 快速提交：`outputs/submissions/results_seg_ep20_fast_scaled.json`

验证指标：

```json
{
  "mAP50": 0.06670703457088263,
  "recall_at_iou50": 0.27941176470588236,
  "tiny_recall_at_iou50": 0.17647058823529413,
  "large_mean_best_iou": 0.2467566599166909,
  "avg_inference_time_ms": 21.310110623560526
}
```

快速提交耗时：

```json
{
  "avg_inference_time_ms": 30.16719601328904,
  "max_inference_time_ms": 484.561
}
```

### YOLO26n-seg 参考权重候选

- 权重来源：`/home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/runs/yolo26n_seg_baseline-2/weights/best.pt`
- 参考项目训练：`epochs=200, imgsz=640, batch=2`
- 参考项目结果：best Box mAP50 `0.66858`，best Mask mAP50 `0.54176`
- 当前工程统一验证：`outputs/reports/val_metrics_seg_ref_yolo26n.json`
- 快速提交：`outputs/submissions/results_seg_ref_yolo26n_fast.json`
- 当前 IoU 优先提交：`outputs/submissions/results_seg_ref_yolo26n_hybrid_unionfloor05.json`
- 当前 mask_box 基础候选提交：`outputs/submissions/results_seg_ref_yolo26n_hybrid_conf001_tiny_w2h6_elong02_union_maskbox.json`
- 当前 mAP 优先候选提交：`outputs/submissions/results_seg_ref_yolo26n_hybrid_union_maskbox_floor020_norm90000.json`
- 当前 mAP/Recall 优先候选提交：`outputs/submissions/results_seg_ref_yolo26n_hybrid_adapt1536_thr4096.json`
- 当前推荐交付包：`deliverables/yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate`

当前工程统一验证指标：

```json
{
  "mAP50": 0.3520961985180195,
  "recall_at_iou50": 0.6823529411764706,
  "tiny_recall_at_iou50": 0.5882352941176471,
  "large_mean_best_iou": 0.15606858284114652,
  "avg_inference_time_ms": 22.056523350371837
}
```

快速提交耗时：

```json
{
  "avg_inference_time_ms": 33.14948504983388,
  "max_inference_time_ms": 518.99
}
```

## 2. 当前最佳候选

当前最佳候选是 YOLO26n-seg 参考权重 + `configs/yolo_seg_crack_hybrid.yaml` 混合推理模式，其中 `infer.conf=0.01`。后处理同时启用大面积预测框扩张、极窄小框补偿、长条框方向扩张和长裂纹跨框合并：`box_expand_ratio=0.12, box_expand_pixels=12, box_expand_min_area=90000, tiny_box_min_width=2, tiny_box_min_height=6, tiny_box_max_area=80, tiny_box_min_side=6, tiny_box_max_width=8, elongated_box_expand_ratio=0.2, elongated_box_min_area=30000, elongated_box_min_aspect=3, union_elongated_clusters=true, union_cluster_box_source=mask_box, union_cluster_score_factor=0.2, union_cluster_score_floor=0.2, union_cluster_score_area_norm=90000`。

选择理由：

- 在当前统一验证口径下，mAP50、Recall、tiny Recall 均明显高于 20 epoch YOLO11n-seg 基准。
- hybrid 提交格式合法，长度为 301。
- 验证集提交口径 mAP50 高于 fast 和 tile 两种策略；当前面积相关 union 分数下限进一步提高了 mAP50 和 Large Matched IoU。
- 测试集平均推理耗时约 61.2ms，最大单图约 1424ms，满足 2s 级单图速度约束。
- 已生成 `.pth` 权重副本和复现交付包。

验证集提交口径指标：

```json
{
  "mAP50": 0.529718,
  "precision": 0.181474,
  "recall_at_iou50": 0.8470588235294118,
  "tiny_recall_at_iou50": 0.9411764705882353,
  "large_mean_matched_iou": 0.747287,
  "large_mean_best_iou": 0.816483
}
```

主要不足：

- `large_mean_best_iou=0.8165`，极大裂纹定位精度接近但仍未达到 0.85。
- 新增 `union_cluster_box_source=mask_box` 后，`large_mean_best_iou` 从 `0.7976` 提升到 `0.8165`；继续加入 `union_cluster_score_floor=0.2, union_cluster_score_area_norm=90000` 后，`large_mean_matched_iou` 从 `0.6854` 提升到 `0.7473`。这说明大裂纹好框已存在，但仍有部分样本受候选质量或排序影响，尚未达到 `0.85` 目标。
- `tiny_recall_at_iou50=0.9412`，当前验证集已超过 90% 目标。
- 低阈值带来更多候选框，`precision_at_conf=0.1962`，误检风险高于 `conf=0.04`。
- 全切片模式能提升 tiny recall 和 large IoU，但最大耗时略超 2s，需要继续优化触发策略。

## 3. 提交后处理口径对比

`validate_yolo.py` 直接评估 YOLO 模型输出框；`eval_submission.py` 评估最终提交 JSON 中的 `predict_bboxes`，更接近实际提交口径。以下结果均使用 YOLO26n-seg 参考权重，在验证集 257 张图上评估。

| 策略 | 提交预测 | mAP50 | Recall | Tiny Recall | Large Best IoU | Avg Time | Max Time |
|---|---|---:|---:|---:|---:|---:|---:|
| fast 缩放整图 | `val_pred_seg_ref_yolo26n_fast.json` | 0.4090 | 0.6588 | 0.4706 | 0.4699 | 27.65ms | 451.50ms |
| hybrid 条件切片 | `val_pred_seg_ref_yolo26n_hybrid.json` | 0.4169 | 0.6794 | 0.4706 | 0.4699 | 30.24ms | 440.46ms |
| hybrid conf=0.04 | `val_pred_seg_ref_yolo26n_hybrid_conf004.json` | 0.4391 | 0.7088 | 0.6471 | 0.5247 | 32.14ms | 459.10ms |
| hybrid conf=0.04 + large expand 0.12/12 | `val_pred_seg_ref_yolo26n_hybrid_conf004_expand_large012_px12.json` | 0.4535 | 0.7206 | 0.6471 | 0.6097 | 约32ms | 约459ms |
| hybrid conf=0.03 + large expand 0.12/12 | `val_pred_seg_ref_yolo26n_hybrid_conf003_expand_large012_px12.json` | 0.4683 | 0.7353 | 0.6471 | 0.6097 | 未单独统计 | 未单独统计 |
| hybrid conf=0.02 + large expand 0.12/12 | `val_pred_seg_ref_yolo26n_hybrid_conf002_expand_large012_px12.json` | 0.4886 | 0.7647 | 0.7059 | 0.6107 | 未单独统计 | 未单独统计 |
| hybrid conf=0.01 + large expand 0.12/12 | `val_pred_seg_ref_yolo26n_hybrid_conf001_expand_large012_px12.json` | 0.5053 | 0.7794 | 0.8235 | 0.6173 | 未单独统计 | 未单独统计 |
| hybrid conf=0.005 + large expand 0.12/12 | `val_pred_seg_ref_yolo26n_hybrid_conf0005_expand_large012_px12.json` | 0.5120 | 0.7882 | 0.8235 | 0.6216 | 未单独统计 | 未单独统计 |
| hybrid conf=0.01 + tile large | `val_pred_seg_ref_yolo26n_hybrid_conf001_tilelarge_expand_large012_px12.json` | 0.4797 | 0.7853 | 0.8235 | 0.6415 | 更慢 | 更慢 |
| hybrid conf=0.01 + large expand + tiny w2h6 | `val_pred_seg_ref_yolo26n_hybrid_conf001_expand_large012_px12_tiny_w2h6.json` | 0.5086 | 0.7853 | 0.9412 | 0.6173 | 未单独统计 | 未单独统计 |
| hybrid conf=0.01 + tiny w2h6 + elong 0.2 | `val_pred_seg_ref_yolo26n_hybrid_conf001_expand_large012_px12_tiny_w2h6_elong02_area30000.json` | 0.5089 | 0.7882 | 0.9412 | 0.6337 | 未单独统计 | 未单独统计 |
| hybrid conf=0.01 + tiny w2h6 + elong 0.2 + edge 80 | `val_pred_seg_ref_yolo26n_hybrid_conf001_tiny_w2h6_elong02_edge80.json` | 0.5089 | 0.7882 | 0.9412 | 0.6337 | 未单独统计 | 未单独统计 |
| hybrid conf=0.01 + tiny w2h6 + elong 0.2 + edge 200 | `val_pred_seg_ref_yolo26n_hybrid_conf001_tiny_w2h6_elong02_edge200.json` | 0.5081 | 0.7882 | 0.9412 | 0.6292 | 未单独统计 | 未单独统计 |
| hybrid conf=0.01 + tiny w2h6 + elong 0.2 + tile large | `val_pred_seg_ref_yolo26n_hybrid_conf001_tiny_w2h6_elong02_tilelarge.json` | 0.4832 | 0.7912 | 0.9412 | 0.6453 | 更慢 | 更慢 |
| hybrid conf=0.01 + tiny w2h6 + elong 0.2 + union score0.2 | `val_pred_seg_ref_yolo26n_hybrid_conf001_tiny_w2h6_elong02_union_score02.json` | 0.5292 | 0.8471 | 0.9412 | 0.7976 | 未单独统计 | 未单独统计 |
| hybrid conf=0.01 + union score0.2 + union source prefer_mask | `val_pred_seg_ref_yolo26n_hybrid_conf001_tiny_w2h6_elong02_union_prefermask.json` | 0.5291 | 0.8471 | 0.9412 | 0.8165 | 未单独统计 | 未单独统计 |
| hybrid conf=0.01 + union score0.2 + union source mask_box | `val_pred_seg_ref_yolo26n_hybrid_conf001_tiny_w2h6_elong02_union_maskbox.json` | 0.5291 | 0.8471 | 0.9412 | 0.8165 | 未单独统计 | 未单独统计 |
| hybrid conf=0.01 + mask_box + union score floor0.2 area90000 | `val_pred_seg_ref_yolo26n_hybrid_union_maskbox_floor020_norm90000.json` | 0.5297 | 0.8471 | 0.9412 | 0.8165 | 未单独统计 | 未单独统计 |
| tile 切片融合 | `val_pred_seg_ref_yolo26n_tile.json` | 0.4019 | 0.6941 | 0.5882 | 0.5151 | 81.36ms | 2568.23ms |

结论：

- `hybrid conf=0.01 + mask_box + union floor0.2 area90000` 在 mAP50、Recall、Tiny Recall、Large Matched IoU、Large Best IoU 和速度之间取得当前最好平衡，适合作为当前推荐提交模式。
- `union_cluster_box_source=mask_box/prefer_mask` 基本保持 mAP50、Recall 和 Tiny Recall，同时把 Large Best IoU 从 `0.7976` 提升到 `0.8165`。该改动通过保留原始 mask 外接框作为 union 簇合并依据，减少扩张框背景对大裂纹合并的干扰。
- `union_cluster_score_floor=0.2, union_cluster_score_area_norm=90000` 让大 union 框具备面积相关最低分数，缓解“大裂纹高 IoU 好框分数太低，被低 IoU 高分框抢先匹配”的问题；当前验证 mAP50 和 Large Matched IoU 均优于旧推荐，作为新的推荐提交参数。
- `conf=0.005` 的验证 mAP50 略高，但预测框数量和误检风险更高，当前不作为主提交。
- `tile large` 的 Large Best IoU 更高，但验证 mAP50 更低且速度更慢，适合作为后续专项优化方向。
- `edge anchor` 没有带来有效收益，暂不启用。
- 对长裂纹 union 框做横向厚度鲁棒裁剪的离线诊断没有提升平均 Large Best IoU：基础 best IoU 为 `0.6337`，普通候选 union 的 oracle 均值约 `0.7574`，鲁棒裁剪 `trim=0.15` 约 `0.7037`、`trim=0.25` 约 `0.6337`，因此当前不启用裁剪式 union。
- `fast` 的速度也达标，适合作为保底提交模式。
- `tile` 的 Recall、Tiny Recall、Large Best IoU 更好，但最大耗时略超 2s，需要进一步做条件触发或减少切片数量。
- 后续不应只看 YOLO 原生验证表，还应以 `eval_submission.py` 的提交后处理口径作为主要决策依据。

测试集提交耗时：

| 策略 | 提交文件 | Avg Time | Max Time |
|---|---|---:|---:|
| fast | `results_seg_ref_yolo26n_fast.json` | 33.15ms | 518.99ms |
| hybrid | `results_seg_ref_yolo26n_hybrid.json` | 37.23ms | 624.76ms |
| hybrid conf=0.04 | `results_seg_ref_yolo26n_hybrid_conf004.json` | 41.85ms | 1092.42ms |
| hybrid conf=0.04 + large expand 0.12/12 | `results_seg_ref_yolo26n_hybrid_conf004_expand_large012_px12.json` | 43.29ms | 1110.46ms |
| hybrid conf=0.01 + large expand 0.12/12 | `results_seg_ref_yolo26n_hybrid_conf001_expand_large012_px12.json` | 61.03ms | 1437.19ms |
| hybrid conf=0.01 + large expand + tiny w2h6 | `results_seg_ref_yolo26n_hybrid_conf001_expand_large012_px12_tiny_w2h6.json` | 61.05ms | 1463.84ms |
| hybrid conf=0.01 + tiny w2h6 + elong 0.2 | `results_seg_ref_yolo26n_hybrid_conf001_expand_large012_px12_tiny_w2h6_elong02_area30000.json` | 61.50ms | 1475.15ms |
| hybrid conf=0.01 + tiny w2h6 + elong 0.2 + union score0.2 | `results_seg_ref_yolo26n_hybrid_conf001_tiny_w2h6_elong02_union_score02.json` | 61.17ms | 1423.56ms |
| hybrid conf=0.01 + union score0.2 + union source mask_box | `results_seg_ref_yolo26n_hybrid_conf001_tiny_w2h6_elong02_union_maskbox.json` | 60.99ms | 1460.53ms |
| hybrid conf=0.01 + mask_box + union floor0.2 area90000 | `results_seg_ref_yolo26n_hybrid_union_maskbox_floor020_norm90000.json` | 60.08ms | 1431.24ms |

### Union 框源消融

本轮新增 `union_cluster_box_source` 参数，控制细长裂纹簇合并时使用哪一种框：

- `box`：使用最终扩张后的提交框，保持旧逻辑。
- `mask_box`：使用分割 mask 的原始外接框，更贴近裂纹像素。
- `prefer_mask`：有 `mask_box` 时优先使用，否则回退到 `box`。

验证集 257 张图结果：

| box source | mAP50 | Precision | Recall | Tiny Recall | Large Matched IoU | Large Best IoU | Pred Boxes |
|---|---:|---:|---:|---:|---:|---:|---:|
| box | 0.529181 | 0.181360 | 0.847059 | 0.941176 | 0.685062 | 0.797588 | 1587 |
| prefer_mask | 0.529143 | 0.181132 | 0.847059 | 0.941176 | 0.685408 | 0.816483 | 1590 |
| mask_box | 0.529148 | 0.181474 | 0.847059 | 0.941176 | 0.685408 | 0.816483 | 1587 |
| mask_box + floor0.2 area90000 | 0.529718 | 0.181474 | 0.847059 | 0.941176 | 0.747287 | 0.816483 | 1587 |
| mask_box + floor0.3 area90000 | 0.528745 | 0.181474 | 0.847059 | 0.941176 | 0.747287 | 0.816483 | 1587 |
| mask_box + floor0.4 area90000 | 0.527610 | 0.181474 | 0.847059 | 0.941176 | 0.748360 | 0.816483 | 1587 |
| mask_box + floor0.5 area90000 | 0.524657 | 0.181474 | 0.847059 | 0.941176 | 0.769792 | 0.816483 | 1587 |
| mask_box + floor0.6 area90000 | 0.520934 | 0.181474 | 0.847059 | 0.941176 | 0.777128 | 0.816483 | 1587 |

结论：`mask_box` 和 `prefer_mask` 对整体 mAP50 与 Recall 几乎无损，对 Large Best IoU 有正收益；提高 `union_cluster_score_floor` 不新增任何候选框，但能让大裂纹 union 框更早参与全局 score 贪心匹配。`floor0.5 area90000` 将 Large Matched IoU 从 `0.7473` 提升到 `0.7698`，mAP50 从 `0.5297` 降到 `0.5247`，适合作为大裂纹定位参考消融；`floor0.2` 更偏 mAP 优先。

### 大框形状变体消融

诊断文件：

```text
outputs/reports/large_box_shape_diagnostics_maskbox_floor020_norm90000.csv
outputs/reports/large_iou_diagnosis_current.csv
outputs/reports/large_iou_diagnosis_current.json
outputs/reports/large_iou_diagnosis_largevar_combo004.csv
outputs/reports/large_iou_diagnosis_largevar_combo004.json
```

诊断结论：

- 当前验证集中 large GT 共 16 个。
- 低 IoU 大框主要有三类误差：框太窄或太短导致覆盖不足、框过大导致背景过多、中心偏移。
- 一些样本已经存在高 IoU 预测框，但分数低；`union_cluster_score_floor=0.2` 已缓解这类排序问题。
- 当前推荐提交的 large failure 类型：`shape_quality_gap=9`、`good_candidate_low_score=4`、`ok=3`。
- `combo004` 大框变体把 `large_mean_best_iou` 提高到 `0.8432`，但 failure 类型变为 `good_candidate_not_matched=2`、`shape_quality_gap=7`、`good_candidate_low_score=4`、`ok=3`。这说明新变体确实产生了更贴近 GT 的框，但这些框分数通常只有约 `0.05`，没有在全局 score 贪心匹配中排到低 IoU 高分框之前。

新增可选后处理：

```yaml
large_box_variants: false
large_variant_expand_ratio: 0.0
large_variant_shrink_ratio: 0.0
large_variant_directional_expand_ratio: 0.0
```

该策略会为大面积预测框额外生成少量扩张/收缩候选，不替换原框。当前默认关闭。

验证集消融：

| 策略 | mAP50 | Precision | Recall | Tiny Recall | Large Matched IoU | Large Best IoU | Pred Boxes |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | 0.529718 | 0.181474 | 0.847059 | 0.941176 | 0.747287 | 0.816483 | 1587 |
| expand006 | 0.528485 | 0.176147 | 0.847059 | 0.941176 | 0.747287 | 0.832364 | 1635 |
| shrink006 | 0.528485 | 0.176147 | 0.847059 | 0.941176 | 0.747287 | 0.826809 | 1635 |
| dir006 | 0.525549 | 0.162437 | 0.847059 | 0.941176 | 0.747287 | 0.842341 | 1773 |
| combo004 | 0.523928 | 0.154589 | 0.847059 | 0.941176 | 0.747287 | 0.843228 | 1863 |
| dir006 floor0.2 max12 | 0.512554 | 0.163358 | 0.847059 | 0.941176 | 0.747633 | 0.841608 | 1763 |
| union expand006 floor0.2 | 0.526640 | 0.178660 | 0.847059 | 0.941176 | 0.747287 | 0.826610 | 1612 |
| union dir006 floor0.2 | 0.518842 | 0.170819 | 0.847059 | 0.941176 | 0.747287 | 0.834386 | 1686 |
| union dir010 floor0.2 | 0.518842 | 0.170819 | 0.847059 | 0.941176 | 0.747287 | 0.838453 | 1686 |
| union combo004 floor0.2 | 0.514534 | 0.165994 | 0.847059 | 0.941176 | 0.747287 | 0.835481 | 1735 |
| combo004 floor0.3 max8 | 0.503503 | 0.164854 | 0.847059 | 0.941176 | 0.738204 | 0.835973 | 1747 |
| combo004 floor0.5 max8 | 0.472157 | 0.164854 | 0.847059 | 0.941176 | 0.739444 | 0.835973 | 1747 |
| combo004 floor0.8 max8 | 0.363494 | 0.164854 | 0.847059 | 0.941176 | 0.745261 | 0.835973 | 1747 |
| union combo004 floor0.5 max8 | 0.478782 | 0.167345 | 0.847059 | 0.941176 | 0.759439 | 0.835240 | 1721 |
| union combo004 floor0.8 max8 | 0.378059 | 0.167345 | 0.847059 | 0.941176 | 0.777838 | 0.835240 | 1721 |

结论：大框变体可以把 Large Best IoU 提高到约 `0.8432`，接近 `0.85` 参考值，说明框形状修正方向有效；但新增候选数量增加后 mAP50 和 Precision 下降，Large Matched IoU 基本不变。进一步限制为 union-only 变体后，新增框更少；当把 union 变体分数下限提高到 `0.8` 时，Large Matched IoU 可提升到 `0.7778`，但 mAP50 会下降到 `0.3781`，不适合作为总分最优提交。

高分变体失败原因：当前推荐提交在验证集有 `39` 个预测框 `score>=0.8`、`205` 个预测框 `score>=0.5`；`union combo004 floor0.8 max8` 会把 `score>=0.8` 的预测增加到 `173` 个、`score>=0.5` 增加到 `339` 个，导致 AP 早期排序被大量变体候选占据。因此当前不启用 `large_box_variants` 作为默认提交策略，也不生成新的测试提交，只保留为后续大目标专项优化工具。

复现诊断命令：

```bash
python src/diagnose_large_iou.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --submit outputs/submissions/val_pred_seg_ref_yolo26n_hybrid_union_maskbox_floor020_norm90000.json \
  --split val \
  --out-csv outputs/reports/large_iou_diagnosis_current.csv \
  --out-json outputs/reports/large_iou_diagnosis_current.json
```

### 大图全局分支与切片触发消融

本组实验测试 `global_max_side` 和 `tile_trigger=large` 是否能在不使用高分变体的情况下改善整体 mAP、Recall 或大目标定位。

验证集结果：

| 策略 | mAP50 | Precision | Recall | Tiny Recall | Large Matched IoU | Large Best IoU | Pred Boxes |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | 0.529718 | 0.181474 | 0.847059 | 0.941176 | 0.747287 | 0.816483 | 1587 |
| global1536 | 0.531683 | 0.184783 | 0.850000 | 0.941176 | 0.733837 | 0.819037 | 1564 |
| global1792 | 0.533073 | 0.182449 | 0.850000 | 0.941176 | 0.731215 | 0.795729 | 1584 |
| adapt1536 thr4096 | 0.531683 | 0.184783 | 0.850000 | 0.941176 | 0.733837 | 0.819037 | 1564 |
| adapt1792 thr2560 | 0.530666 | 0.181818 | 0.847059 | 0.941176 | 0.745261 | 0.793025 | 1584 |
| adapt1792 thr3072 | 0.533073 | 0.182449 | 0.850000 | 0.941176 | 0.731215 | 0.795729 | 1584 |
| adapt1792 thr4096 | 0.533073 | 0.182449 | 0.850000 | 0.941176 | 0.731215 | 0.795729 | 1584 |
| tilelarge maxtiles4 | 0.507105 | 0.153153 | 0.850000 | 0.941176 | 0.723805 | 0.811434 | 1887 |
| tilelarge maxtiles6 | 0.502218 | 0.143924 | 0.850000 | 0.941176 | 0.731310 | 0.814730 | 2008 |
| global1536 tilelarge4 | 0.506743 | 0.155329 | 0.852941 | 0.941176 | 0.718951 | 0.798944 | 1867 |

测试集 `global1792` 提交：

```text
outputs/submissions/results_seg_ref_yolo26n_hybrid_global1792.json
```

提交合法，但耗时超出目标：

```json
{
  "avg_inference_time_ms": 78.0749,
  "max_inference_time_ms": 2554.494,
  "large_avg_ms": 480.4415,
  "large_max_ms": 2554.494
}
```

结论：`global1792` 在验证集 mAP50 最高，Recall 也从 `0.8471` 提升到 `0.8500`，但 Large Matched IoU 下降，且测试集最大耗时超过 2s；因此只作为高 mAP 离线对照，不替代当前主提交。强制大图切片会增加预测框数量并降低 mAP，暂不启用。

自适应全图分支候选：

```text
outputs/submissions/results_seg_ref_yolo26n_hybrid_adapt1536_thr4096.json
```

命令：

```bash
python src/infer_submit_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --weights /home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/runs/yolo26n_seg_baseline-2/weights/best.pt \
  --split test \
  --adaptive-global-max-side 1536 \
  --adaptive-global-threshold 4096 \
  --out outputs/submissions/results_seg_ref_yolo26n_hybrid_adapt1536_thr4096.json
```

耗时：

```json
{
  "avg_inference_time_ms": 65.6273,
  "max_inference_time_ms": 1447.567,
  "regular_avg_ms": 45.0155,
  "large_avg_ms": 327.0229,
  "large_max_ms": 1447.567
}
```

结论：`adapt1536_thr4096` 满足 2s 最大耗时，验证 mAP50 和 Recall 高于当前推荐，且 Large Best IoU 略高；但 Large Matched IoU 低于当前推荐。因此它适合作为“mAP/Recall 优先候选”，不替代当前 “Large Matched IoU 优先”主提交。

### 当前推荐提交选择

IoU 优先主提交：

```text
outputs/submissions/results_seg_ref_yolo26n_hybrid_unionfloor05.json
```

验证：

```text
mAP50=0.524657
Recall=0.847059
Tiny Recall=0.941176
Large Matched IoU=0.769792
Large Best IoU=0.816483
```

测试耗时：

```text
avg=61.59ms
max=1457.59ms
large_avg=259.92ms
large_max=1444.93ms
```

mAP 优先候选：

```text
outputs/submissions/results_seg_ref_yolo26n_hybrid_union_maskbox_floor020_norm90000.json
```

mAP/Recall 优先候选：

```text
outputs/submissions/results_seg_ref_yolo26n_hybrid_adapt1536_thr4096.json
```

## 4. 后续优化方向

1. 正式长训当前工程 YOLO-seg：使用 `imgsz=1024/1280` 和 200 epoch，比较是否超过参考权重。
2. 训练侧尺度感知重采样：已新增 `src/build_scale_aware_train_list.py` 并生成 `data/yolo_seg/crack_seg_scaleaware.yaml`。基础训练图 1028 张，重复后训练列表 1274 行；tiny 图 51 张、large 图 64 张、huge 图 43 张被提高采样频率。下一步需要完成 200 epoch 长训并比较硬指标。
3. 训练侧局部尺度 crop：已新增 `src/build_scale_crop_dataset.py` 并生成 `data/yolo_seg/crack_seg_scalecrop.yaml`。原始 train 1028 行，新增 crop 187 张，合并训练列表 1215 行；其中 tiny crop 120 张、large crop 67 张，标签文件均非空。下一步需要完成 200 epoch 长训并比较硬指标。
4. 组合训练清单：已新增 `src/build_combined_yolo_data.py` 并生成 `data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml`。该清单合并 `train_scaleaware.txt` 的 1274 行整图重采样和 `train_scalecrop_only.txt` 的 187 行局部 crop，总计 1461 行、1215 个唯一图像/crop；缺失图片 0，缺失标签 0，空标签 0。推荐作为下一轮 200 epoch 长训候选。
5. 长训前数据预检：已新增 `src/check_yolo_seg_data.py`，并对 `data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml` 运行通过。结果为 `ok=True`，train 1461 行、unique 1215、duplicates 246，val 257，test 301，failures 为空。duplicates 是有意过采样，不作为错误。
6. 组合数据 smoke train：已用 `data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml` 完成 1 epoch 链路测试。环境为 torch 2.3.1+cu121、Ultralytics 8.4.63、RTX 4060 Ti。训练扫描 train 1461 张、val 257 张，0 corrupt；生成 `runs/crack_yolo_seg/yolo11n_seg_scalecombo_smoke/weights/best.pt`、`last.pt`、`args.yaml`、`results.csv` 和曲线图。该 run 仅验证训练入口，不作为正式指标。
7. 阶段化流水线：已新增 `scripts/run_pipeline.py`，支持 `prepare`、`check`、`smoke`、`train`、`eval-ref`、`submit-ref`、`package` 阶段。已验证 `--stages prepare check smoke --dry-run` 和真实 `--stages check`，默认不会误启动 200 epoch 长训。
8. 大目标优化：继续改进大图全局分支和 union 框形状。大框变体已能把 Large Best IoU 推到约 0.843，但会降低 mAP；`global1792` 能提高 mAP 但超出 2s 最大耗时。下一步应尝试按图像尺寸自适应 global side，例如只对 `max_side<4096` 的大图使用 1792，对超大图保持 1280。
9. 小目标优化：针对 `tiny` 样本做重采样，验证低 `conf`、更高 `imgsz`、有限切片策略对 tiny recall 的提升。
10. 速度优化：保留 `configs/yolo_seg_crack_fast.yaml` 作为速度保底，另建 `hybrid` 模式只对小目标高风险图触发切片。
11. 消融实验：对比 detect baseline、YOLO11n-seg、YOLO26n-seg、fast/global/tile 三种推理模式。

## 5. 离线大框后处理搜索

为验证“只靠提交后处理追加大框变体”是否能把极大裂纹 `mean bbox IoU` 推到 0.85，本轮新增脚本：

```text
src/search_large_box_postprocess.py
```

该脚本不重新跑模型，只读取已有验证集提交 JSON，给大面积或细长预测框追加少量扩张、收缩、平移候选框，然后用 `eval_submission.py` 同一套提交口径计算 `mAP50`、`Recall`、`Tiny Recall`、`Large Matched IoU` 和 `Large Best IoU`。

快速搜索命令：

```bash
python src/search_large_box_postprocess.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --submit outputs/submissions/val_pred_seg_ref_yolo26n_hybrid_unionfloor05.json \
  --split val \
  --out-dir outputs/large_box_search/unionfloor05_quick \
  --expand-x 0 0.08 0.16 \
  --expand-y 0 0.08 0.16 \
  --shrink-x 0 0.08 \
  --shrink-y 0 0.08 \
  --shift-x 0 \
  --shift-y 0 \
  --score-factor 0.25 0.5 \
  --score-floor 0.2 0.5 0.8 \
  --max-new-per-image 8
```

快速搜索最佳结果：

```text
Large Matched IoU: 0.7719
Large Best IoU:    0.8287
mAP50:             0.4421
Recall:            0.8471
Tiny Recall:       0.9412
```

定向搜索命令：

```bash
python src/search_large_box_postprocess.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --submit outputs/submissions/val_pred_seg_ref_yolo26n_hybrid_unionfloor05.json \
  --split val \
  --out-dir outputs/large_box_search/unionfloor05_focused \
  --expand-x 0 0.08 0.16 \
  --expand-y 0 0.08 0.16 \
  --shrink-x 0 0.08 0.16 \
  --shrink-y 0 0.08 0.16 \
  --shift-x -0.04 0 0.04 \
  --shift-y 0 \
  --score-factor 0.2 \
  --score-floor 0.5 0.8 \
  --max-new-per-image 3
```

定向搜索最佳结果：

```text
参数：expand_x=0.08, expand_y=0.0, shrink_x=0.0, shrink_y=0.0, shift_x=-0.04, score_floor=0.8
Large Matched IoU: 0.7751
Large Best IoU:    0.8290
mAP50:             0.4703
Recall:            0.8471
Tiny Recall:       0.9412
```

结论：通用大框扩张、收缩、平移候选只能把 Large Matched IoU 从 `0.7698` 提升到 `0.7751`，距离 `0.85` 仍很远，并且会明显降低 mAP50。因此该方向作为消融保留，不替代当前推荐提交。

基于诊断 CSV 的失败类型，极大目标瓶颈主要不是简单分数排序，而是模型没有稳定产生足够贴合 GT 的候选框。后续更可能有效的方向：

1. 训练侧增强：对极大裂纹样本重采样，加入大图局部/全局双尺度训练，而不是只在推理端修框。
2. 模型侧增强：使用更大 YOLO-seg 模型或专门的大目标分支，提高极大裂纹整体 mask 质量。
3. 推理侧增强：对大图采用更高分辨率全图分支，但需要 TensorRT/ONNX 或更严格 tile 数控制以满足 2s。
4. 标签侧分析：逐张可视化 `images/1998.png`、`images/1997.png`、`images/670.bmp` 等低 IoU 大目标，确认 GT bbox 是否覆盖超长弱纹理或存在 mask/bbox 标注不一致。

## 6. 大裂纹低 IoU 可视化诊断

为支持逐张分析极大裂纹 IoU 不达标的原因，本轮新增：

```text
src/visualize_large_iou_cases.py
```

该脚本读取 `src/diagnose_large_iou.py` 生成的诊断 CSV，并在验证集原图上同时绘制 GT bbox、matched 预测框和 best-IoU 预测框。

颜色约定：

- 绿色：GT bbox
- 红色：按提交匹配到该 GT 的 matched 预测框
- 蓝色：该 GT 的 best-IoU 预测框

运行命令：

```bash
python src/visualize_large_iou_cases.py \
  --dataset dataset \
  --diagnosis-csv outputs/reports/large_iou_diagnosis_current.csv \
  --out-dir outputs/visualizations/large_iou_cases_current \
  --limit 12 \
  --max-side 1600
```

已生成：

```text
outputs/visualizations/large_iou_cases_current/index.json
outputs/visualizations/large_iou_cases_current/01_id2199_1998_miou0.544.jpg
outputs/visualizations/large_iou_cases_current/02_id698_670_miou0.559.jpg
outputs/visualizations/large_iou_cases_current/03_id1418_1374_miou0.585.jpg
...
outputs/visualizations/large_iou_cases_current/12_id537_509_miou0.824.jpg
```

代表性诊断：

| image_id | image | failure_type | matched IoU | best IoU | 现象 |
|---:|---|---|---:|---:|---|
| 2199 | images/1998.png | shape_quality_gap | 0.5438 | 0.5438 | 当前模型没有产生足够贴合的大框候选 |
| 698 | images/670.bmp | shape_quality_gap | 0.5593 | 0.6231 | best 框高度覆盖不足，定位形状缺口明显 |
| 1418 | images/1374.png | good_candidate_low_score | 0.5848 | 0.9277 | 好框存在但分数低，排序匹配被高分差框抢占 |
| 1077 | images/1049.bmp | good_candidate_low_score | 0.6658 | 0.9249 | 好框存在但分数低 |
| 824 | images/796.bmp | good_candidate_low_score | 0.7162 | 0.9119 | 好框存在但分数低 |

结论：可视化进一步确认极大裂纹问题分为两类。一类是好框存在但排序分数低，可继续优化 union/variant 的分数策略；另一类是 best 框本身也低，必须依赖训练侧大目标增强、更高分辨率全图分支或更强模型来提升候选质量。

## 7. 常规图推理速度优化实验

为降低 `max_side<=2048` 常规图的单张耗时，本轮在 `src/infer_submit_seg.py` 中新增两个可调推理参数：

```yaml
infer:
  direct_resize_max_side: 1280
  keep_masks_for_merge: true
```

参数含义：

- `direct_resize_max_side`：常规图仍按 `direct_max_side` 判定为整图分支，但当最大边超过该值时，先缩放到该最大边推理，再把 mask 外接框坐标映射回原图。目标是减少接近 2048 图像的分割 mask 后处理耗时。
- `keep_masks_for_merge`：是否保留全图 mask 做 mask IoU 融合。设为 `false` 时只使用 mask 外接框和 bbox IoU 合并，速度更快，但会损失 mask 级融合带来的召回。

已运行候选对比：

| 候选 | 验证 mAP50 | Recall | Tiny Recall | Large Matched IoU | 测试 avg ms | 常规 avg ms | 常规 max ms | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 原始 unionfloor05 | 0.5247 | 0.8471 | 0.9412 | 0.7698 | 61.59 | 45.95 | 1457.59 | 指标最好，常规单张最慢 |
| `direct_resize_max_side=1280` | 0.5247 | 0.8471 | 0.9412 | 0.7698 | 56.98 | 41.08 | 456.18 | 平均速度改善，指标不变 |
| `direct_resize_max_side=960` | 0.5247 | 0.8471 | 0.9412 | 0.7698 | 55.67 | 40.00 | 461.35 | 平均速度进一步改善，指标不变 |
| `keep_masks_for_merge=false` | 0.4166 | 0.8176 | 0.8235 | 0.7680 | 未作为最终候选 | 未作为最终候选 | 未作为最终候选 | Tiny Recall 大幅下降，否决 |

结论：

1. `direct_resize_max_side=960/1280` 对当前验证集预测指标没有负面影响，并能降低测试集平均耗时。
2. 常规图严格单张 `<100ms` 仍未达标，慢样本不只来自接近 2048 的大图，也来自预测框较多的中小 BMP 图，后处理和 mask 清理仍是瓶颈。
3. 关闭全图 mask 融合会明显损失 Tiny Recall，不适合作为最终提交策略。
4. 下一步速度优化应优先做“只保留 mask_box 但改进 bbox 合并质量”的轻量融合，或导出 ONNX/TensorRT 后在目标 4080 环境复测；模型侧优先训练更强候选以提升 mAP50、Recall 和速度稳定性。

## 8. Detect Head 快速框分支实验

为进一步降低分割 mask 后处理耗时，本轮在 `src/infer_submit_seg.py` 中新增：

```yaml
infer:
  prediction_box_source: mask_box   # mask_box / det_box / prefer_mask
  retina_masks: true
```

命令行可覆盖：

```bash
python src/infer_submit_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --weights /home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/runs/yolo26n_seg_baseline-2/weights/best.pt \
  --split val \
  --prediction-box-source det_box \
  --no-retina-masks \
  --direct-resize-max-side 960 \
  --out outputs/submissions/val_pred_seg_ref_yolo26n_hybrid_detbox_fast.json
```

验证结果：

| 候选 | mAP50 | Recall | Tiny Recall | Large Matched IoU | Large Best IoU | 结论 |
|---|---:|---:|---:|---:|---:|---|
| mask_box 主候选 | 0.5247 | 0.8471 | 0.9412 | 0.7698 | 0.8165 | 当前推荐 |
| det_box + no_retina 快速分支 | 0.4221 | 0.7412 | 0.8824 | 0.2488 | 0.4729 | 否决 |

结论：该模型的 detect head bbox 与裂纹真实 bbox 对齐不足，尤其极大裂纹定位严重下降；当前任务仍应使用 mask 外接框作为提交 bbox 来源。

## 9. Scale-aware + Scale-crop 正式长训

为进一步提升 mAP50、Recall 和小裂纹召回，当前已启动训练侧优化，使用组合训练清单：

```text
data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml
```

训练启动脚本：

```text
scripts/train_scalecombo_200e.sh
```

运行中的 run 目录：

```text
runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop
```

训练参数：

```text
model=/home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/yolo11n-seg.pt
data=data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml
imgsz=1024
epochs=200
batch=2
device=0
```

启动时检查：

```text
data check ok=True
train rows=1461
train unique=1215
val rows=257
test rows=301
GPU=NVIDIA RTX 4060 Ti 8GB
```

训练已开始写入：

```text
runs/.../args.yaml
runs/.../results.csv
runs/.../weights/best.pt
runs/.../weights/last.pt
```

监控命令：

```bash
RUN=runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop
tail -n 5 "$RUN/results.csv"
python scripts/monitor_training.py --run-dir "$RUN" --epochs 200
nvidia-smi
tensorboard --logdir runs/crack_yolo_seg --host 0.0.0.0 --port 6006
```

训练完成后评估：

```bash
python src/infer_submit_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --weights "$RUN/weights/best.pt" \
  --split val \
  --direct-resize-max-side 960 \
  --out outputs/submissions/val_pred_scalecombo_best.json

python src/eval_submission.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --submit outputs/submissions/val_pred_scalecombo_best.json \
  --split val \
  --out outputs/reports/submission_metrics_scalecombo_best_val.json \
  --errors outputs/reports/submission_errors_scalecombo_best_val.csv
```

或直接运行一键评估打包脚本：

```bash
bash scripts/eval_package_scalecombo.sh \
  runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop \
  yolo11n_seg_scalecombo_best_candidate
```

该脚本会依次生成：

```text
outputs/submissions/val_pred_yolo11n_seg_scalecombo_best_candidate.json
outputs/reports/submission_metrics_yolo11n_seg_scalecombo_best_candidate_val.json
outputs/submissions/results_yolo11n_seg_scalecombo_best_candidate.json
outputs/reports/inference_time_summary_yolo11n_seg_scalecombo_best_candidate.json
outputs/reports/speed_buckets_yolo11n_seg_scalecombo_best_candidate_test.json
deliverables/yolo11n_seg_scalecombo_best_candidate/
outputs/reports/delivery_audit_yolo11n_seg_scalecombo_best_candidate.md
```
