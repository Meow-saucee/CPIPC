# 验证划分稳定性说明

## 结论先行

`seed=42` 是工程里为了可复现而固定的常见默认随机种子，不是根据验证集分数反复挑出来的种子。

但是，当前正式报告主要基于 `seed=42, val_ratio=0.2` 的单次 train/val 划分。因此它能说明该划分上的本地验证表现，不能单独证明模型在所有 trainval 内部划分上都稳定，也不能排除某个划分相对容易的可能性。

## 当前划分方式

划分逻辑在 `src/prepare_yolo.py::stratified_split()`：

- 先按图像是否含 tiny 裂纹、large 裂纹、normal 裂纹分组。
- 每组内部按随机种子 shuffle。
- 每组抽约 20% 作为 val。
- 这样能避免 val 集完全缺少 tiny 或 large 样本。

当前固化文件：

```text
data/yolo_seg/split_manifest.yaml
```

当前设置：

```text
seed: 42
val_ratio: 0.2
train_count: 1028
val_count: 257
test_count: 301
```

## 不能直接做的错误验证

不能只把 `trainval` 重新抽一批 val，然后用已经在 `seed=42` train split 上训练好的同一个模型去评估。

原因是：新的 val 里大概率包含旧 train 里的图片。模型已经见过这些图片，这会造成训练泄漏，指标会虚高，不能证明泛化稳定。

## 更严谨的验证方式

推荐做 repeated holdout：

1. 固定同一套模型、训练参数和后处理参数。
2. 选择多个 seed，例如 `0,1,2,3,4,42`。
3. 每个 seed 重新生成 train/val。
4. 每个 seed 独立训练一遍模型。
5. 分别在各自未参与训练的 val 上评估。
6. 汇总 mAP50、Recall、Tiny Recall、Large IoU 的均值、最小值、最大值和波动范围。

可执行但成本较高，因为 200 epoch 单次训练约数小时。

## 新增工具

脚本：

```text
src/evaluate_split_stability.py
```

只审计不同 seed 的 val 分布：

```bash
python src/evaluate_split_stability.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --dataset dataset \
  --seeds 0,1,2,3,4,42 \
  --out-json outputs/reports/split_stability_summary.json \
  --out-csv outputs/reports/split_stability_summary.csv
```

当前已实际运行过一次分布审计，输出文件：

```text
outputs/reports/split_stability_summary.json
outputs/reports/split_stability_summary.csv
```

结果摘要：

| seed | val images | GT boxes | tiny boxes | large boxes | 与 seed=42 val 重叠数 | Jaccard |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 257 | 347 | 14 | 18 | 53 | 0.115 |
| 1 | 257 | 333 | 16 | 17 | 54 | 0.117 |
| 2 | 257 | 314 | 15 | 17 | 51 | 0.110 |
| 3 | 257 | 327 | 14 | 16 | 45 | 0.096 |
| 4 | 257 | 313 | 13 | 16 | 39 | 0.082 |
| 42 | 257 | 340 | 17 | 16 | 257 | 1.000 |

这个结果说明：分层策略能让各个 val 都包含 tiny 和 large 样本，但不同 seed 的 GT 总数和样本集合差异明显。因此，单一 `seed=42` 指标不应被表述为跨划分稳定性结论。

如果已经有覆盖完整 `trainval` 的预测 JSON，可进一步评估这些划分的指标：

```bash
python src/evaluate_split_stability.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --dataset dataset \
  --seeds 0,1,2,3,4,42 \
  --submission outputs/submissions/trainval_pred_full.json \
  --out-json outputs/reports/split_stability_summary_with_metrics.json \
  --out-csv outputs/reports/split_stability_summary_with_metrics.csv
```

注意：这个指标模式只适合评估“没有训练泄漏”的预测。例如每个 seed 对应各自重新训练模型的 val 预测，或专门设计的 out-of-fold 预测。单个 `seed=42` 模型对其他 seed 的 val 评估不严谨。

## 当前应如何表述

建议在答辩或报告中这样说：

```text
当前使用 seed=42 的分层 holdout 作为固定本地验证集，保证 tiny、large、normal 样本在 val 中都有覆盖，并保证实验可复现。该结果不是多 seed 重复验证结论。若需要证明指标对划分不敏感，应进一步做 repeated holdout 或 k-fold，并报告 mAP50/Recall/Tiny Recall 的均值和方差。
```
