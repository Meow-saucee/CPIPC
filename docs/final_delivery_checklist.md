# 最终交付清单

本文用于提交前核对“代码、权重、结果、报告、复现说明”是否齐全。

## 1. 推荐最终方案

```text
当前保底候选：yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate
当前训练候选：yolo11n_seg_scalecombo_best_candidate
当前综合推荐候选：ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate
模型路线：YOLO26n-seg 实例分割训练 / mask 转 bbox 提交
推理策略：全局缩放 + 低预测数条件切片 + bbox/mask 融合 + 大框条件扩张 + 极窄小框补偿 + 长条框方向扩张 + 长裂纹跨框合并 + mask_box union 框源 + 面积相关 union score floor
配置文件：configs/yolo_seg_crack_hybrid.yaml
关键阈值：infer.conf=0.01
```

说明：`ensemble_y26_y11_weighted_candidate` 是验证 mAP50 与召回优先基线；`ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate` 在 weighted 基线上对 24 张 regular 慢图使用 fast-detbox 分支，作为当前速度合规优先最终候选；`ensemble_y26_y11_w075_calibrated_demote_candidate` 是大裂纹定位参考指标更高的备选。当前不再把 Large IoU 0.85 作为当前提交选择的硬门槛。最终提交候选以对应审计报告为准。

## 2. 核心交付包

```text
deliverables/yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate/
```

包内文件：

| 文件 | 用途 |
|---|---|
| `weights/yolo26n_ref_unionfloor05.pth` | ensemble 第一分支权重，高 tiny recall |
| `weights/yolo11n_scalecombo_best.pth` | ensemble 第二分支权重，200 epoch scalecombo 模型 |
| `weights/ensemble_y26_y11_weighted_candidate.pth` | 主权重副本，保留用于单模型加载检查 |
| `results.json` | 测试集最终提交结果 |
| `configs/yolo_seg_crack_hybrid.yaml` | 最终推理配置 |
| `reports/val_metrics.json` | 验证集提交口径指标 |
| `reports/val_errors.csv` | 验证集错误分析 |
| `reports/inference_time_summary.json` | 测试集推理耗时统计 |
| `reports/speed_buckets.json` | 常规图/大图分桶耗时 |
| `docs/model_system_architecture.md` | 输入输出、模型框架和系统架构图 |
| `docs/model_framework_and_parameters.md` | 模型架构、输入输出、关键参数说明 |
| `docs/model_architecture_overview.md` | 一页式模型输入输出、网络结构、推理后处理和改参入口 |
| `docs/parameter_map.md` | 关键参数地图，说明改哪些参数会影响召回、IoU、mAP 和速度 |
| `docs/experiment_summary.md` | 实验消融和候选提交对比 |
| `README.md` | 交付包说明 |
| `REPRODUCE.md` | 复现实验说明 |
| `manifest.json` | 交付包文件索引 |
| `source/src/` | 完整 Python 源码副本 |
| `source/scripts/reproduce_final_speed_route.sh` | 复现最终 speed-route 提交 |
| `source/scripts/reproduce_ensemble_weighted.sh` | 从两套权重复现 weighted ensemble 基线 |
| `source/configs/` | 所有训练/推理配置副本 |
| `source/requirements.txt` | pip 依赖说明 |
| `source/environment.yml` | Conda 环境说明 |

## 3. 工程源码

核心源码位于：

```text
src/
configs/
docs/
README.md
requirements.txt
environment.yml
```

最终交付包内也包含源码副本：

```text
deliverables/yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate/source/src/
deliverables/yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate/source/configs/
deliverables/yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate/source/requirements.txt
deliverables/yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate/source/environment.yml
```

当前综合推荐 ensemble 包：

```text
deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/
```

复现入口：

```bash
bash scripts/reproduce_final_speed_route.sh
```

主要入口：

| 脚本 | 入口函数 | 用途 |
|---|---|---|
| `src/data_analyze.py` | `main()` | 数据集统计 |
| `src/prepare_yolo.py` | `main()` | bbox 检测数据转换 |
| `src/prepare_yolo_seg.py` | `main()` | mask 转 YOLO-seg |
| `src/train_yolo.py` | `main()` | YOLO 检测训练 |
| `src/train_yolo_seg.py` | `main()` | YOLO-seg 训练 |
| `src/validate_yolo.py` | `main()` | 检测模型验证 |
| `src/infer_submit.py` | `main()` | 检测模型推理提交 |
| `src/infer_submit_seg.py` | `main()` | 分割模型推理提交 |
| `src/eval_submission.py` | `main()` | 按提交 JSON 口径评估验证集 |
| `src/diagnose_large_iou.py` | `main()` | 大裂纹 matched/best/top-score 诊断 |
| `src/check_submit.py` | `main()` | 提交文件合法性检查 |
| `src/summarize_submission_time.py` | `main()` | 推理耗时统计 |
| `src/visualize_predictions.py` | `main()` | GT/预测可视化 |
| `src/package_delivery.py` | `main()` | 打包交付目录 |
| `scripts/run_pipeline.py` | `main()` | 端到端阶段化复现入口 |

## 4. 技术报告与答辩材料

| 文件 | 内容 |
|---|---|
| `docs/technical_design_report.md` | 正式技术设计报告 |
| `docs/model_system_architecture.md` | 模型系统架构与关键参数 |
| `docs/model_architecture_overview.md` | 一页式模型架构总览和参数入口 |
| `docs/parameter_map.md` | 输入输出、训练、推理、后处理、评估参数速查 |
| `docs/experiment_summary.md` | 已运行实验、指标与耗时 |
| `docs/defense_slides_outline.md` | 答辩 PPT 提纲 |
| `docs/final_delivery_checklist.md` | 本交付核对表 |

## 5. 当前已验证结果

### 5.1 保底候选

验证集提交口径：

| 指标 | 数值 |
|---|---:|
| images | 257 |
| ground truth boxes | 340 |
| predicted boxes | 1588 |
| mAP50 | 0.5247 |
| Precision | 0.1814 |
| Recall@IoU50 | 0.8471 |
| Tiny Recall@IoU50 | 0.9412 |
| Large Matched IoU | 0.7698 |
| Large Best IoU | 0.8165 |

测试集提交合法性：

```text
Submission is valid: outputs/submissions/results_seg_ref_yolo26n_hybrid_unionfloor05.json, rows=301
```

测试集推理耗时：

| 指标 | 数值 |
|---|---:|
| images | 301 |
| avg inference time | 61.59ms |
| max inference time | 1457.59ms |
| min inference time | 9.13ms |

另有 mAP/Recall 优先候选：

```text
outputs/submissions/results_seg_ref_yolo26n_hybrid_adapt1536_thr4096.json
验证 mAP50=0.5317, Recall=0.8500, Tiny Recall=0.9412, Large Matched IoU=0.7338
测试耗时 avg=65.63ms, max=1447.57ms
```

### 5.2 当前综合推荐候选

```text
候选名称：ensemble_weighted_route_regular_gt100_fastdetbox768_warm
提交文件：outputs/submissions/results_ensemble_weighted_route_regular_gt100_fastdetbox768_warm.json
交付包：deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/
审计报告：outputs/reports/delivery_audit_ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate.md
```

验证集提交口径：

| 指标 | 数值 |
|---|---:|
| mAP50 | 0.5765 |
| Recall@IoU50 | 0.9147 |
| Tiny Recall@IoU50 | 0.9412 |
| Large Matched IoU | 0.7981 |
| Large Best IoU | 0.8356 |

说明：该候选按 mAP50/Recall 优先选择 weighted ensemble，再对 24 张 regular 慢图使用 fast-detbox768 分支满足速度约束。若需要更高的大裂纹定位参考指标，可切换到 `ensemble_y26_y11_w075_calibrated_demote_candidate`。

速度审计：

| 项目 | 结果 |
|---|---|
| regular avg | 30.62ms，达标 |
| regular max | 93.838ms，达标 |
| large max | 1444.930ms，达标 |

补充检查显示，忽略首张 warmup 后仍有 regular 图 `images/1455.bmp` 约 1457ms，因此速度问题不是单纯预热，而是个别接近 2048 边界的 regular 图在双模型融合流程下过慢。

已测试简单尺寸路由：regular 图用 yolo11n、large 图用 w075 calibrated ensemble。该策略会把 tiny recall 降到 0.7647，因此不采用。当前仍以 weighted ensemble 作为 mAP50/召回优先候选。

### 5.3 已训练的新候选

```text
候选名称：yolo11n_seg_scalecombo_best_candidate
训练目录：runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop
训练数据：data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml
训练命令：python src/train_yolo_seg.py --config configs/yolo_seg_crack_hybrid.yaml --data-yaml data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml --model /home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/yolo11n-seg.pt --imgsz 1024 --epochs 200 --batch 2 --device 0 --tag seg-scaleaware-scalecrop
```

训练完成后自动生成：

```text
outputs/reports/submission_metrics_yolo11n_seg_scalecombo_best_candidate_val.json
outputs/reports/speed_buckets_yolo11n_seg_scalecombo_best_candidate_test.json
outputs/reports/delivery_audit_yolo11n_seg_scalecombo_best_candidate.md
outputs/reports/candidate_comparison.md
outputs/submissions/results_yolo11n_seg_scalecombo_best_candidate.json
deliverables/yolo11n_seg_scalecombo_best_candidate/
```

最终选择时优先看 `outputs/reports/candidate_comparison.md` 中的对比字段：

| 字段 | 目标 |
|---|---|
| `mAP50` | 主排名参考指标 |
| `recall_at_iou50` | 漏检控制指标 |
| `tiny_recall_at_iou50` | 极小裂纹召回指标 |
| `large_mean_matched_iou` | 大裂纹定位参考指标，不再作为 0.85 硬门槛 |
| `regular_speed_pass` | 常规图单张最大耗时 < 100ms 的速度检查 |
| `large_speed_pass` | 超大图单张最大耗时 < 2000ms 的速度检查 |

## 6. 复现命令

### 6.1 环境

```bash
cd /home/ruiyi/CPIPC/Dection
conda activate cpipc-crack
```

如需重建环境：

```bash
conda env create -f environment.yml
conda activate cpipc-crack
```

### 6.2 数据转换

```bash
python src/prepare_yolo_seg.py --config configs/yolo_seg_crack_hybrid.yaml
```

### 6.3 验证集评估

```bash
python src/infer_submit_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --weights /home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/runs/yolo26n_seg_baseline-2/weights/best.pt \
  --split val \
  --union-cluster-score-floor 0.5 \
  --union-cluster-score-area-norm 90000 \
  --out outputs/submissions/val_pred_seg_ref_yolo26n_hybrid_unionfloor05.json

python src/eval_submission.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --submit outputs/submissions/val_pred_seg_ref_yolo26n_hybrid_unionfloor05.json \
  --split val \
  --out outputs/reports/submission_metrics_seg_ref_yolo26n_hybrid_unionfloor05_val.json
```

### 6.4 测试集提交

```bash
python src/infer_submit_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --weights /home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/runs/yolo26n_seg_baseline-2/weights/best.pt \
  --split test \
  --union-cluster-score-floor 0.5 \
  --union-cluster-score-area-norm 90000 \
  --out outputs/submissions/results_seg_ref_yolo26n_hybrid_unionfloor05.json

python src/check_submit.py \
  --dataset dataset \
  --submit outputs/submissions/results_seg_ref_yolo26n_hybrid_unionfloor05.json

python src/summarize_submission_time.py \
  --submit outputs/submissions/results_seg_ref_yolo26n_hybrid_unionfloor05.json \
  --out-json outputs/reports/inference_time_summary_seg_ref_yolo26n_hybrid_unionfloor05.json \
  --out-csv outputs/reports/inference_time_per_image_seg_ref_yolo26n_hybrid_unionfloor05.csv
```

### 6.5 可视化

```bash
python src/visualize_predictions.py \
  --dataset dataset \
  --split test \
  --submit outputs/submissions/results_seg_ref_yolo26n_hybrid_unionfloor05.json \
  --limit 20
```

## 7. 提交前检查命令

```bash
python -m compileall src

python src/check_submit.py \
  --dataset dataset \
  --submit outputs/submissions/results_seg_ref_yolo26n_hybrid_unionfloor05.json

python src/summarize_submission_time.py \
  --submit outputs/submissions/results_seg_ref_yolo26n_hybrid_unionfloor05.json

python src/audit_delivery.py \
  --delivery deliverables/yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate \
  --dataset dataset \
  --out-json outputs/reports/delivery_audit_unionfloor05_strict.json \
  --out-md outputs/reports/delivery_audit_unionfloor05_strict.md
```

交付包校验结果：

```text
python -m compileall src
Submission is valid: deliverables/yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate/results.json, rows=301
source files copied: 25
.pth weight exists: yes
REPRODUCE.md exists: yes
technical_design_report.md exists: yes
Delivery audit status: FAIL
```

审计脚本输出：

| 文件 | 内容 |
|---|---|
| `outputs/reports/delivery_audit_unionfloor05_strict.json` | 机器可读验收结果 |
| `outputs/reports/delivery_audit_unionfloor05_strict.md` | 人工复查报告 |

当前严格审计未通过项：

| 检查项 | 当前证据 | 影响 |
|---|---|---|
| `regular_image_single_speed_target` | 常规图最大耗时 `1457.589ms > 100ms` | 接近 2048 边界的 BMP 常规图仍有单张测速风险 |

## 8. 尚未完全达成的目标

当前方案已经形成完整工程和合法提交，但以下性能目标仍需关注：

| 目标 | 当前值 | 状态 |
|---|---:|---|
| Tiny Recall ≥ 90% | 94.12% | 当前验证集达标 |
| Large bbox IoU | 作为定位质量参考 | 不再作为必须继续优化到 0.85 的硬目标 |
| 超大图 < 2s | max 1457.59ms | 当前达标 |
| 常规图 < 100ms | avg 45.95ms, max 1457.59ms | 平均达标，最大值未达标 |

后续应优先围绕 mAP50、极小裂纹召回和常规图单张速度继续优化；大裂纹定位作为诊断指标保留。
