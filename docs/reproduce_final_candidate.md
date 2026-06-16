# 最终候选复现实验运行手册

本文记录当前正在训练的 scale-aware + scale-crop YOLO-seg 候选，从训练产物到验证、测试推理、交付包、审计和 TensorBoard 查看的一套复现流程。

## 1. 当前候选信息

```text
候选名：yolo11n_seg_scalecombo_best_candidate
模型路线：YOLO11n-seg 实例分割
训练数据：data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml
基础配置：configs/yolo_seg_crack_hybrid.yaml
输入尺寸：imgsz=1024
训练轮数：epochs=200
批大小：batch=2
随机种子：seed=42
实验标签：seg-scaleaware-scalecrop
```

训练目录：

```text
runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop
```

训练命令：

```bash
cd /home/ruiyi/CPIPC/Dection
conda activate cpipc-crack

python src/train_yolo_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --data-yaml data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml \
  --model /home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/yolo11n-seg.pt \
  --imgsz 1024 \
  --epochs 200 \
  --batch 2 \
  --device 0 \
  --tag seg-scaleaware-scalecrop
```

## 2. 训练监控

```bash
cd /home/ruiyi/CPIPC/Dection
conda activate cpipc-crack

RUN=runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop

python scripts/monitor_training.py --run-dir "$RUN" --epochs 200
nvidia-smi
```

重点查看：

- 当前 epoch 是否达到 200。
- `weights/best.pt` 和 `weights/last.pt` 是否存在。
- `results.csv` 是否记录完整训练曲线。
- `best_box` 和 `best_mask` 只是 Ultralytics 验证指标，不等同于最终比赛提交指标。

也可以保存机器可读的训练进度：

```bash
python scripts/monitor_training.py \
  --run-dir "$RUN" \
  --epochs 200 \
  --json-out outputs/reports/training_progress_yolo11n_scalecombo.json
```

## 3. TensorBoard 查看

训练过程中或训练结束后均可查看：

```bash
cd /home/ruiyi/CPIPC/Dection
conda activate cpipc-crack

tensorboard \
  --logdir runs/crack_yolo_seg \
  --host 0.0.0.0 \
  --port 6006
```

浏览器打开：

```text
http://localhost:6006
```

如需查看归档后的实验：

```bash
tensorboard --logdir experiments --host 0.0.0.0 --port 6006
```

## 4. 训练完成后的评估、推理和打包

训练完成后执行：

```bash
cd /home/ruiyi/CPIPC/Dection
conda activate cpipc-crack

RUN=runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop

bash scripts/eval_package_scalecombo.sh "$RUN" yolo11n_seg_scalecombo_best_candidate
```

如果不想手动盯训练进度，可以使用自动等待脚本。它会等 `results.csv` 达到指定 epoch 后，自动运行验证、测试推理、打包、审计、候选对比和实验归档：

```bash
cd /home/ruiyi/CPIPC/Dection
conda activate cpipc-crack

python scripts/wait_eval_scalecombo.py \
  --run-dir runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop \
  --name yolo11n_seg_scalecombo_best_candidate \
  --epochs 200 \
  --train-epochs 200 \
  --poll-seconds 60
```

当前已启动的自动等待进程：

```text
PID: 183510
日志: logs/wait_eval_scalecombo.log
自动进度 JSON: outputs/reports/training_progress_yolo11n_seg_scalecombo_best_candidate.json
```

查看方式：

```bash
ps -p 183510 -o pid,etime,cmd
tail -f logs/wait_eval_scalecombo.log
```

该脚本会依次生成：

```text
outputs/submissions/val_pred_yolo11n_seg_scalecombo_best_candidate.json
outputs/reports/submission_metrics_yolo11n_seg_scalecombo_best_candidate_val.json
outputs/reports/submission_errors_yolo11n_seg_scalecombo_best_candidate_val.csv
outputs/submissions/results_yolo11n_seg_scalecombo_best_candidate.json
outputs/reports/inference_time_summary_yolo11n_seg_scalecombo_best_candidate.json
outputs/reports/inference_time_per_image_yolo11n_seg_scalecombo_best_candidate.csv
outputs/reports/speed_buckets_yolo11n_seg_scalecombo_best_candidate_test.json
deliverables/yolo11n_seg_scalecombo_best_candidate/
outputs/reports/delivery_audit_yolo11n_seg_scalecombo_best_candidate.json
outputs/reports/delivery_audit_yolo11n_seg_scalecombo_best_candidate.md
```

## 5. 实验归档

打包后可额外把训练 run 归档到 `experiments/`，保存规范命名的 `best_epoch`、`last_epoch`、曲线和 TensorBoard 文件：

```bash
cd /home/ruiyi/CPIPC/Dection
conda activate cpipc-crack

RUN=runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop

python src/archive_experiment.py \
  --run-dir "$RUN" \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --data-yaml data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml \
  --model yolo11n-seg.pt \
  --dataset-name cpipc-chip-crack-seg \
  --imgsz 1024 \
  --epochs 200 \
  --batch 2 \
  --seed 42 \
  --tag seg-scaleaware-scalecrop \
  --metrics outputs/reports/submission_metrics_yolo11n_seg_scalecombo_best_candidate_val.json \
  --errors outputs/reports/submission_errors_yolo11n_seg_scalecombo_best_candidate_val.csv \
  --submission outputs/submissions/results_yolo11n_seg_scalecombo_best_candidate.json \
  --command "python src/train_yolo_seg.py --config configs/yolo_seg_crack_hybrid.yaml --data-yaml data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml --model /home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/yolo11n-seg.pt --imgsz 1024 --epochs 200 --batch 2 --device 0 --tag seg-scaleaware-scalecrop"
```

归档目录示例：

```text
experiments/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop/
```

## 6. 指标验收口径

最终判断以 `audit_delivery.py` 和提交口径验证指标为准，不以 Ultralytics 训练日志单独判断。

重点文件：

```text
outputs/reports/submission_metrics_yolo11n_seg_scalecombo_best_candidate_val.json
outputs/reports/speed_buckets_yolo11n_seg_scalecombo_best_candidate_test.json
outputs/reports/delivery_audit_yolo11n_seg_scalecombo_best_candidate.md
```

重点指标：

```text
mAP50
recall_at_iou50
tiny_recall_at_iou50
large_mean_matched_iou
large_mean_best_iou
avg_inference_time_ms
常规图最大推理耗时
超大图最大推理耗时
```

目标门槛：

```text
Tiny Recall >= 0.90
mAP50 and recall are the primary selection metrics
Large mean matched bbox IoU is kept as a localization diagnostic metric
常规图单张推理耗时 < 100ms
超大图单张推理耗时 < 2000ms
测试提交 JSON 合法，行数等于测试集图片数
交付包包含 .pth 权重、源码、配置、报告和复现说明
```

如果审计未通过，不要把该候选称为最终达标方案；应根据提交格式、mAP50、Recall、Tiny Recall、速度和错误分析继续调参或换模型。当前不再把 Large IoU 0.85 作为必须继续优化的硬门槛。

## 7. 与参考候选对比

完成 `scripts/eval_package_scalecombo.sh` 后，可把新候选和当前参考候选并排比较：

```bash
cd /home/ruiyi/CPIPC/Dection
conda activate cpipc-crack

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
    outputs/reports/delivery_audit_yolo11n_seg_scalecombo_best_candidate.json \
  --out-json outputs/reports/candidate_comparison.json \
  --out-csv outputs/reports/candidate_comparison.csv \
  --out-md outputs/reports/candidate_comparison.md
```

输出文件：

```text
outputs/reports/candidate_comparison.json
outputs/reports/candidate_comparison.csv
outputs/reports/candidate_comparison.md
```

优先比较：

```text
tiny_recall_at_iou50
large_mean_matched_iou
mAP50
regular_max_ms
large_max_ms
audit_status
```
