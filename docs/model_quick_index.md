# 模型系统快速索引

本文件用于快速定位本项目的输入输出、模型框架图、关键参数和实验产物。更完整解释见：

- `docs/model_system_architecture.md`
- `docs/model_framework_and_parameters.md`
- `docs/model_architecture_overview.md`
- `docs/parameter_map.md`
- `docs/experiment_summary.md`
- `docs/next_optimization_plan.md`

## 1. 模型框架图

可直接打开以下 SVG：

- 整体训练/推理流程：`docs/assets/system_pipeline.svg`
- YOLO-seg 网络结构：`docs/assets/yolo_seg_architecture.svg`
- 跨尺度推理与后处理：`docs/assets/inference_postprocess.svg`

也可以用浏览器打开：

```bash
xdg-open docs/assets/system_pipeline.svg
xdg-open docs/assets/yolo_seg_architecture.svg
xdg-open docs/assets/inference_postprocess.svg
```

## 2. 输入与输出

原始输入：

```text
dataset/trainval/images
dataset/trainval/trainval.json
dataset/test/images
dataset/test/test.json
```

训练输入：

```text
data/yolo_seg/crack_seg.yaml
data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml
```

模型输出：

```text
runs/crack_yolo_seg/<experiment_name>/weights/best.pt
runs/crack_yolo_seg/<experiment_name>/weights/last.pt
runs/crack_yolo_seg/<experiment_name>/results.csv
runs/crack_yolo_seg/<experiment_name>/args.yaml
```

比赛提交输出：

```text
outputs/submissions/results*.json
```

## 3. 当前主线模型

当前主线是 YOLO 实例分割：

```text
灰度芯片图 + bbox/segmentation 标注
-> YOLO-seg 训练
-> 输出 bbox、score、class、mask
-> mask 外接框 / bbox 后处理
-> 官方 predict_bboxes JSON
```

推荐理解方式：

- Backbone：提取芯片纹理、裂纹边缘和形态特征。
- Neck：融合 P3/P4/P5 多尺度特征，兼顾小裂纹和大裂纹。
- Detect Head：输出框、置信度和 `crack` 类别。
- Seg Head：输出裂纹 mask，后续转换为 bbox 提交。

## 4. 关键参数修改位置

参数修改总表：

```text
docs/parameter_map.md
```

主配置文件：

```text
configs/yolo_seg_crack_hybrid.yaml
```

常改训练参数：

```yaml
train:
  model: yolo11n-seg.pt
  imgsz: 1024
  epochs: 200
  batch: 2
  patience: 50
  close_mosaic: 20
  mask_ratio: 4
```

常改推理参数：

```yaml
infer:
  imgsz: 1280
  conf: 0.01
  iou: 0.55
  direct_resize_max_side: 1280
  tile_size: 1280
  tile_overlap: 256
  prediction_box_source: mask_box
  retina_masks: true
```

训练增强代码位置：

```text
src/train_yolo_seg.py
```

跨尺度推理和后处理代码位置：

```text
src/infer_submit_seg.py
```

验证指标代码位置：

```text
src/eval_submission.py
```

候选对比代码位置：

```text
src/compare_candidates.py
```

实验归档代码位置：

```text
src/archive_experiment.py
src/experiment_utils.py
```

## 5. 实验命名与归档

实验名格式由 `src/experiment_utils.py::build_experiment_name()` 生成：

```text
模型名_数据集名_img输入尺寸_ep训练轮数_bs批大小_seed随机种子_tag实验标签
```

示例：

```text
yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop
```

训练结束后可归档：

```bash
python src/archive_experiment.py \
  --run-dir runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --tag seg-scaleaware-scalecrop
```

归档后会保存：

```text
experiments/<experiment_name>/checkpoints/*best_epoch*.pt
experiments/<experiment_name>/checkpoints/*last_epoch*.pt
experiments/<experiment_name>/reports/results.csv
experiments/<experiment_name>/reports/args.yaml
experiments/<experiment_name>/tensorboard/
experiments/<experiment_name>/experiment.json
experiments/<experiment_name>/experiment.yaml
```

TensorBoard 查看：

```bash
tensorboard --logdir experiments --host 0.0.0.0 --port 6006
```

## 6. 当前训练监控命令

```bash
cd /home/ruiyi/CPIPC/Dection
conda activate cpipc-crack

RUN=runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop

python scripts/monitor_training.py --run-dir "$RUN" --epochs 200
nvidia-smi
```

训练完成后执行验证、推理、打包和审计：

```bash
bash scripts/eval_package_scalecombo.sh "$RUN" yolo11n_seg_scalecombo_best_candidate
```

也可以自动等待训练完成后继续执行后续流程：

```bash
python scripts/wait_eval_scalecombo.py \
  --run-dir "$RUN" \
  --name yolo11n_seg_scalecombo_best_candidate \
  --epochs 200 \
  --train-epochs 200 \
  --poll-seconds 60
```

比较新候选与参考候选：

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
    outputs/reports/delivery_audit_yolo11n_seg_scalecombo_best_candidate.json \
  --out-json outputs/reports/candidate_comparison.json \
  --out-csv outputs/reports/candidate_comparison.csv \
  --out-md outputs/reports/candidate_comparison.md
```

注意：只有 `eval_package_scalecombo.sh` 和 `audit_delivery.py` 运行结束后，才能判断提交格式、mAP50、Recall、Tiny Recall、速度和大裂纹定位参考指标。当前不再把 Large IoU 0.85 作为必须继续优化的硬门槛。
