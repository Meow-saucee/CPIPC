# 裂纹检测/分割系统关键参数地图

本文用于快速回答两个问题：

1. 模型系统的输入、输出和模块边界是什么。
2. 想调整小裂纹召回、大裂纹 IoU、mAP 或推理速度时，应修改哪些文件和参数。

当前主线是 YOLO-seg 实例分割方案：

```text
dataset/trainval 灰度图 + bbox/segmentation
  -> prepare_yolo_seg.py
  -> data/yolo_seg/*.yaml + YOLO polygon labels
  -> train_yolo_seg.py
  -> best.pt / last.pt / results.csv / TensorBoard
  -> infer_submit_seg.py
  -> mask 外接框 + bbox 后处理
  -> outputs/submissions/results*.json
```

可视化架构图：

- 整体流程：`docs/assets/system_pipeline.svg`
- YOLO-seg 网络：`docs/assets/yolo_seg_architecture.svg`
- 推理后处理：`docs/assets/inference_postprocess.svg`

## 1. 输入输出速查

| 阶段 | 输入 | 输出 | 核心文件 |
| --- | --- | --- | --- |
| 数据转换 | `dataset/trainval/trainval.json`、训练图像 | `data/yolo_seg/labels`、`crack_seg*.yaml` | `src/prepare_yolo_seg.py` |
| 数据增强/扩展 | 原 YOLO-seg 标签 | scale-aware、scale-crop 训练列表 | `src/build_scale_aware_train_list.py`、`src/build_scale_crop_dataset.py` |
| 训练 | `data/yolo_seg/*.yaml`、预训练权重 | `runs/.../weights/best.pt`、`last.pt`、`results.csv` | `src/train_yolo_seg.py` |
| 验证 | `best.pt`、val 图像、GT JSON | mAP50、Recall、tiny recall、large IoU、错误 CSV | `src/eval_submission.py` |
| 测试推理 | `best.pt`、test 图像 | `outputs/submissions/results*.json` | `src/infer_submit_seg.py` |
| 候选融合 | 多个 `results*.json` | 融合后的 `results*.json` | `src/merge_submissions.py` |
| 提交检查 | results JSON、test JSON | 字段、数量、bbox、score 合法性报告 | `src/check_submit.py` |
| 实验归档 | run 目录、评估报告、提交文件 | `experiments/<exp>/` | `src/archive_experiment.py` |

## 2. 训练参数

主位置：`configs/yolo_seg_crack_hybrid.yaml` 的 `train` 段。

| 参数 | 当前值 | 作用 | 何时修改 |
| --- | --- | --- | --- |
| `train.model` | `yolo11n-seg.pt` | 预训练模型结构和初始权重 | 欠拟合或 mAP 上不去时换 `yolo11s-seg.pt`、`yolo11m-seg.pt`；显存紧张时保留 n 模型 |
| `train.imgsz` | `1024` | 训练输入尺寸 | 小裂纹漏检时升到 `1280`；显存不足或速度慢时降到 `768/896` |
| `train.epochs` | `200` | 最大训练轮数 | 曲线未收敛时增加；明显过拟合时减少或依赖早停 |
| `train.batch` | `2` | batch size | 显存允许时增大以稳定训练；OOM 时减小 |
| `train.patience` | `50` | early stopping 耐心 | 指标长时间不提升时自动停；想完整跑满可增大 |
| `train.close_mosaic` | `20` | 最后若干 epoch 关闭 mosaic | 定位不稳时增大；小目标召回低时保留适度 mosaic |
| `train.mask_ratio` | `4` | mask 分支采样比例 | mask 质量差时可调小，但显存和耗时会上升 |
| `split.seed` | `42` | 训练/验证划分随机种子 | 对比实验必须固定；重新划分时修改 |

训练入口 `src/train_yolo_seg.py` 里还有增强参数：

| 参数 | 当前值 | 作用 | 调参方向 |
| --- | --- | --- | --- |
| `degrees` | `90` | 随机旋转 | 芯片方向不固定时保留；定位不稳时降低 |
| `translate` | `0.06` | 平移增强 | 裂纹贴边多时保留；框偏移明显时降低 |
| `scale` | `0.35` | 缩放增强 | 跨尺度泛化需要保留；小目标被缩太小可降低 |
| `fliplr/flipud` | `0.5/0.5` | 水平/垂直翻转 | 通常保留 |
| `mosaic` | `0.8` | 拼接增强 | 提升泛化和小目标出现频率；后期由 `close_mosaic` 关闭 |
| `mixup` | `0.03` | 图像混合 | 过强会影响细裂纹边界，建议小值 |
| `hsv_v` | `0.25` | 灰度亮度扰动 | 背景亮度变化大时保留；误检纹理多时降低 |

常用训练命令：

```bash
python src/train_yolo_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --data-yaml data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml \
  --model /home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/yolo11n-seg.pt \
  --imgsz 1024 --epochs 200 --batch 2 --device 0 \
  --tag seg-scaleaware-scalecrop
```

## 3. 推理参数

主位置：`configs/yolo_seg_crack_hybrid.yaml` 的 `infer` 段。

| 参数 | 当前值 | 作用 | 何时修改 |
| --- | --- | --- | --- |
| `infer.imgsz` | `1280` | 推理输入尺寸 | 小裂纹漏检时增大；速度不达标时降低 |
| `infer.conf` | `0.01` | 置信度阈值 | Recall 不足时降低；误检多时提高 |
| `infer.iou` | `0.55` | NMS IoU 阈值 | 重复框多时降低；同一长裂纹被误删时提高 |
| `infer.max_det` | `300` | 单图最多保留预测数 | 误检多时降低；裂纹密集时提高 |
| `direct_max_side` | `2048` | 常规图整图推理阈值 | 常规图速度超限时降低 |
| `direct_resize_max_side` | `1280` | 常规图最大缩放边 | 速度慢时降低；定位不准时提高 |
| `global_max_side` | `1280` | 大图全局分支缩放边 | 大裂纹 IoU 不足时提高；大图耗时超限时降低 |
| `tile_size` | `1280` | 大图切片尺寸 | 小裂纹漏检时适当减小或保持高分辨率；速度慢时增大或减少切片 |
| `tile_overlap` | `256` | 切片重叠 | 裂纹跨切片断裂时增大；速度慢时减小 |
| `tile_trigger` | `low_preds` | 何时触发切片 | Recall 优先可改 `always`；速度优先保留 `low_preds` |
| `max_tiles` | `8` | 单图最多切片数 | 大图漏检时增大；2s 超限时减小 |
| `prediction_box_source` | `mask_box` | bbox 来源 | mask 质量好时用 `mask_box`；速度优先可试 `det_box` |
| `retina_masks` | `true` | 原图尺度 mask | mask 外接框更准；关闭可加速但可能降 IoU |
| `keep_masks_for_merge` | `true` | 融合时保留 mask | 提升重复框融合质量；关闭可省内存/提速 |

推理入口：`src/infer_submit_seg.py`。

常用推理命令：

```bash
python src/infer_submit_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --weights runs/crack_yolo_seg/<exp>/weights/best.pt \
  --split test \
  --out outputs/submissions/results_<name>.json
```

## 4. 后处理参数

后处理主要服务召回、定位质量和速度三类目标。当前项目不再把 `Large Matched IoU >= 0.85` 作为必须继续优化的硬目标；大裂纹 IoU 保留为定位质量诊断指标。

| 目标 | 参数 | 当前值 | 影响 |
| --- | --- | --- | --- |
| 去重融合 | `box_iou_merge` | `0.5` | 值越低越容易合并重复框 |
| mask 融合 | `mask_iou_merge` | `0.35` | 值越低越容易合并 mask 相近预测 |
| 大框扩张 | `box_expand_ratio` / `box_expand_pixels` | `0.12` / `12.0` | 提升极大 bbox IoU，但可能增大误检框 |
| 大框触发 | `box_expand_min_area` | `90000` | 只对面积足够大的框扩张 |
| 极小框补偿 | `tiny_box_min_width/height` | `2.0/6.0` | 防止极小预测框过窄导致 IoU/Recall 损失 |
| 极小框触发 | `tiny_box_max_area` | `80.0` | 只处理微小预测 |
| 细长框扩张 | `elongated_box_expand_ratio` | `0.2` | 对长裂纹沿长边扩张 |
| 细长框触发 | `elongated_box_min_area/aspect` | `30000/3.0` | 只处理大而细长的裂纹 |
| 长裂纹 union | `union_elongated_clusters` | `true` | 合并被切片切碎的长裂纹 |
| union 间距 | `union_cluster_max_gap` | `256.0` | 越大越容易把断裂预测合并 |
| union 分数 | `union_cluster_score_floor` | `0.5` | 给合并框最低置信度 |

调参建议：

- 极小 Recall 不达标：优先降低 `conf`、增大 `imgsz`、增大 `tile_overlap`，再检查 `tiny_box_*`。
- 极大 IoU 不达标：优先提高 `global_max_side`、开启或加强 `union_elongated_clusters`、微调 `box_expand_*`。
- 常规图推理超过 100ms：降低 `direct_resize_max_side`，减少 `retina_masks` 或改用更小模型。
- 超大图超过 2s：降低 `max_tiles`、使用 `tile_trigger=low_preds`、降低 `global_max_side`。

## 5. 指标与错误分析参数

主位置：`configs/yolo_seg_crack_hybrid.yaml` 的 `eval` 段。

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `eval.iou_match` | `0.5` | 判断 TP/FP/FN 的 IoU 阈值 |
| `eval.tiny_width` | `5` | 极小裂纹宽度阈值 |
| `eval.tiny_area` | `50` | 极小裂纹面积阈值 |
| `eval.large_area` | `90000` | 极大裂纹面积阈值 |

验证命令：

```bash
python src/infer_submit_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --weights runs/crack_yolo_seg/<exp>/weights/best.pt \
  --split val \
  --out outputs/submissions/val_pred_<name>.json

python src/eval_submission.py \
  --dataset dataset \
  --pred outputs/submissions/val_pred_<name>.json \
  --out outputs/reports/submission_metrics_<name>_val.json \
  --errors outputs/reports/submission_errors_<name>_val.csv
```

重点看：

- `mAP50`：整体检测排名相关。
- `recall_at_iou50`：整体召回。
- `tiny_recall_at_iou50`：极小裂纹召回指标。
- `large_mean_matched_iou`：极大裂纹定位参考指标。
- `avg_inference_time_ms`、speed buckets：速度约束。

## 6. 实验命名、ckpt 和 TensorBoard

命名函数：`src/experiment_utils.py::build_experiment_name()`。

格式：

```text
模型名_数据集名_img输入尺寸_ep训练轮数_bs批大小_seed随机种子_tag实验标签
```

示例：

```text
yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop
```

归档命令：

```bash
python src/archive_experiment.py \
  --run-dir runs/crack_yolo_seg/<exp> \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --data-yaml data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml \
  --tag seg-scaleaware-scalecrop
```

归档后会生成：

```text
experiments/<exp>/
  checkpoints/<exp>__best_epochXX_mAP50-XXXX.pt
  checkpoints/<exp>__last_epochYY.pt
  reports/results.csv
  reports/args.yaml
  tensorboard/events.out.tfevents*
  experiment.json
  experiment.yaml
```

TensorBoard 查看：

```bash
tensorboard --logdir experiments --host 0.0.0.0 --port 6006
```

## 7. 常见目标对应修改清单

| 目标 | 优先修改 |
| --- | --- |
| 提高极小裂纹召回 | `train.imgsz`、`infer.imgsz`、`infer.conf`、`tile_size`、`tile_overlap`、scale-crop 数据 |
| 提高极大裂纹 IoU | `global_max_side`、`box_expand_*`、`elongated_box_*`、`union_cluster_*` |
| 提高整体 mAP50 | 模型规模、训练轮数、增强强度、验证集阈值搜索、误检样本清洗 |
| 提高速度 | 小模型、降低 `direct_resize_max_side`、降低 `global_max_side`、限制 `max_tiles`、关闭部分 mask 融合 |
| 控制显存 | 小模型、降低 `imgsz/batch`、减少 `retina_masks`、减少切片数 |
| 保证可复现 | 固定 `seed`、保存 `args.yaml/results.csv/data.yaml/split_manifest.yaml/weights` |

## 8. 候选融合参数

当前综合推荐候选分三种：

- `ensemble_weighted_route_regular_gt100_fastdetbox768_warm`：当前最终提交候选，保留 weighted 验证指标，并通过 fast-detbox 路由满足 regular 图速度。
- `ensemble_y26_y11_weighted`：mAP50/Recall 最优质量基线，但测试集 regular 最大耗时超 100ms。
- `ensemble_y26_y11_w075_calibrated_demote`：大裂纹定位参考指标更高，是定位更稳的备选候选。

最终提交复现入口：

```bash
bash scripts/reproduce_final_speed_route.sh
```

weighted 质量基线入口：

```bash
bash scripts/reproduce_ensemble_weighted.sh
```

手动融合命令：

```bash
python src/merge_submissions.py \
  --inputs outputs/submissions/results_seg_ref_yolo26n_hybrid_unionfloor05.json \
           outputs/submissions/results_yolo11n_seg_scalecombo_best_candidate.json \
  --out outputs/submissions/results_ensemble_y26_y11_weighted.json \
  --iou-thr 0.65 \
  --mode weighted \
  --max-preds 300 \
  --dataset dataset
```

| 参数 | 当前值 | 作用 | 调参方向 |
| --- | --- | --- | --- |
| `--iou-thr` | `0.65` | 融合时判断重复框的 IoU 阈值 | 重复框多时降低；不同裂纹被误合并时提高 |
| `--mode` | `weighted` | 重复框坐标融合方式 | `weighted` 当前综合最好；`higher_score` 更保守；`union` 更偏召回 |
| `--score-scale` | 无 | 对不同模型输出分数加权 | 想让某模型排序更靠前时设置，如 `1.0 0.8` |
| `--max-preds` | `300` | 单图最多保留框数 | 误检多时降低；漏检多时提高 |
| `--dataset` | `dataset` | 读取图像尺寸，clip bbox 防越界 | 提交前必须保留 |

当前验证结果：

```text
ensemble_y26_y11_weighted:
  mAP50 = 0.5765
  Recall@IoU50 = 0.9147
  Tiny Recall@IoU50 = 0.9412
  Large Matched IoU = 0.7981

ensemble_y26_y11_w075_calibrated:
  mAP50 = 0.5555
  Recall@IoU50 = 0.9147
  Tiny Recall@IoU50 = 0.9412
  Large Matched IoU = 0.8318
  Large Best IoU = 0.8706
```

大裂纹定位参考指标更高备选候选的关键参数：

```text
merge_submissions.py:
  --iou-thr 0.75
  --mode weighted

calibrate_boxes.py:
  --scale-x 1.08
  --scale-y 1.04
  --long-scale 0.96
  --short-scale 0.95
```

注意：融合方案已通过提交格式检查。后续优化优先级不再是把大裂纹 IoU 推到 0.85，而是优先保证提交格式、mAP50、召回、极小裂纹召回和推理耗时；大裂纹 IoU 用于辅助判断框定位是否偏大、偏小或排序异常。

速度审计补充：

```text
weighted 质量基线 regular max = 1457.589ms，未满足单张 <100ms。
最终 speed-route 候选 regular max = 93.838ms，regular avg = 30.62ms，large max = 1444.93ms。
该路由只替换 24 张历史 regular 慢图，避免全量 fast-detbox 导致 Tiny Recall 明显下降。
```

后续速度优化方向：

- 为 regular 图增加单模型快速路径，只在大图或低置信预测时启用 ensemble。
- 对 `max_side` 接近 2048 的图降低 `direct_resize_max_side`。
- 将两模型推理改成进程内连续加载并复用模型，减少重复初始化和 I/O 计时。
- 若比赛计时严格按单张端到端，优先提交单模型或 TensorRT/ONNX 加速版本。

已尝试但不推荐的路由：

```text
regular 图使用 yolo11n，large 图使用 w075 calibrated ensemble。
```

验证集结果显示，简单尺寸路由会使 tiny recall 从 0.9412 降到 0.7647，违反最优先的极小裂纹召回目标。因此当前不采用该路由。

交付包中必须包含两套实际参与融合的权重：

```text
deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/weights/yolo26n_ref_unionfloor05.pth
deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/weights/yolo11n_scalecombo_best.pth
```
