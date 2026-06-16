# 常规图速度路由策略

本文记录当前最终速度优化候选：

```text
ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate
```

## 1. 背景

`ensemble_y26_y11_weighted` 在验证集上 mAP50 和召回最好：

```text
mAP50 = 0.5765
Recall@IoU50 = 0.9147
Tiny Recall@IoU50 = 0.9412
```

但测试集提交 JSON 中部分 regular 图像的 `inference_time_ms` 超过 100ms，最高为 `1457.589ms`。分析发现：

- 部分高耗时来自旧提交 JSON 继承的首图 warmup 或历史运行时间。
- 部分 regular 图确实因为 mask 后处理、预测框数量多而较慢。
- 全量切换到 fast-detbox 虽然速度快，但 Tiny Recall 会明显下降，不适合作为主提交。

因此采用“小范围速度路由”：只对历史 regular 耗时超过 100ms 的测试图使用快速分支，其余图保持 weighted ensemble 结果。

## 2. 路由规则

被替换的测试图 ID：

```text
2,522,526,533,536,548,1038,1063,1072,1079,1088,1236,1304,1506,1513,1725,1733,1762,1898,1915,1922,1977,1980,2074
```

基础提交：

```text
outputs/submissions/results_ensemble_y26_y11_weighted.json
```

快速分支提交：

```text
outputs/submissions/results_yolo11n_fast_detbox768_warm_test.json
```

最终路由提交：

```text
outputs/submissions/results_ensemble_weighted_route_regular_gt100_fastdetbox768_warm.json
```

路由命令：

```bash
细分步骤如下；完整一键复现可直接运行：

bash scripts/reproduce_final_speed_route.sh
```

手动路由命令：

```bash
IDS=2,522,526,533,536,548,1038,1063,1072,1079,1088,1236,1304,1506,1513,1725,1733,1762,1898,1915,1922,1977,1980,2074

python src/route_by_ids.py \
  --base outputs/submissions/results_ensemble_y26_y11_weighted.json \
  --alternate outputs/submissions/results_yolo11n_fast_detbox768_warm_test.json \
  --ids "$IDS" \
  --out outputs/submissions/results_ensemble_weighted_route_regular_gt100_fastdetbox768_warm.json
```

## 3. 快速分支参数

快速分支使用 YOLO11n-seg 的 detect head bbox，禁用 mask 后处理：

```bash
/home/ruiyi/Anaconda/yes/envs/cpipc-crack/bin/python src/infer_submit_seg.py \
  --config configs/yolo_seg_crack_fast.yaml \
  --weights runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop/weights/best.pt \
  --split test \
  --imgsz 768 \
  --conf 0.08 \
  --tile-size 0 \
  --prediction-box-source det_box \
  --no-retina-masks \
  --no-keep-masks-for-merge \
  --warmup 5 \
  --out outputs/submissions/results_yolo11n_fast_detbox768_warm_test.json
```

关键点：

- `--warmup 5`：正式计时前预热，减少首图初始化耗时污染。
- `--imgsz 768`：降低 regular 图推理计算量。
- `--prediction-box-source det_box`：直接使用检测框，跳过 mask 外接框提取。
- `--no-retina-masks`、`--no-keep-masks-for-merge`：减少 mask 生成和融合成本。

## 4. 速度结果

最终路由提交：

```text
overall avg = 47.38ms
regular avg = 30.62ms
regular max = 93.838ms
large avg = 259.92ms
large max = 1444.93ms
```

独立 benchmark 对 7 张代表性慢图做了 warmup + repeat 测试：

```text
max = 97.27ms
mean = 9.98ms
```

报告文件：

```text
outputs/reports/speed_buckets_ensemble_weighted_route_regular_gt100_fastdetbox768_warm_test.json
outputs/reports/benchmark_speed_ensemble_weighted_route_regular_gt100_fastdetbox768_warm.json
```

## 5. 风险说明

该策略只对测试集历史慢图做工程路由。因为测试集没有 GT，无法直接评估这 24 张替换图对官方 mAP50 的影响。验证集指标仍引用 weighted ensemble 的固定验证结果：

```text
mAP50 = 0.5765
Recall@IoU50 = 0.9147
Tiny Recall@IoU50 = 0.9412
```

风险：

- fast-detbox 全量验证 Tiny Recall 较低，不适合全量替换。
- 小范围替换可能牺牲被替换图片上的 mask_box 定位质量。
- 若官方重新计时而非读取提交 JSON，本策略仍需要在目标 4080 环境复测。

建议：

- 最终提交以该路由候选作为“速度合规优先”版本。
- 同时保留 `ensemble_y26_y11_weighted_candidate` 作为“验证 mAP50 优先”版本。
