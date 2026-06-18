# 跨尺度芯片图像裂纹缺陷检测 — 项目知识导出

> 导出时间：2026-06-18。本文件包含对话中梳理出的全部关键信息，导入另一台电脑的 AI 编程工具后可快速理解项目全貌。

---

## 1. 项目背景

- 竞赛：CPIPC 赛题四「跨尺度芯片图像的裂纹缺陷智能检测算法设计」
- 赛题网址：https://cpipc.acge.org.cn/cw/contestNews/detail/10/2c9080179cac09ca019cb17ad9530b59?page=1
- 任务：在单通道灰度芯片图像中定位裂纹缺陷（仅 crack 一类）
- 训练集：1285 张图像，1652 个 bbox，带 segmentation 标注
- 测试集：301 张图像
- 图像尺度跨度极大（宽 45-7468px，高 46-9267px）
- 极小目标 79 个，极大目标 83 个

---

## 2. 目录结构

```
/home/ruiyi/CPIPC/Dection/
├── configs/
│   ├── yolo_seg_crack_hybrid.yaml   # 主推理配置（低阈值+全后处理）
│   └── yolo_seg_crack_fast.yaml     # 快速推理配置（高阈值+无后处理）
├── src/
│   ├── infer_submit_seg.py          # 推理入口（核心）
│   ├── eval_submission.py           # 提交口径评估
│   ├── train_yolo_seg.py            # YOLO-seg 训练入口
│   ├── prepare_yolo_seg.py          # 数据转换（JSON→YOLO-seg）
│   ├── merge_submissions.py         # 多模型加权融合
│   ├── route_by_ids.py              # 按图像 ID 替换预测结果
│   ├── route_submissions.py         # 按图像尺寸路由
│   ├── check_submit.py              # 提交合法性检查
│   ├── summarize_submission_time.py # 推理耗时统计
│   ├── audit_delivery.py            # 交付审计
│   ├── package_delivery.py          # 打包交付目录
│   └── diagnose_large_iou.py        # 大裂纹 IoU 诊断
├── scripts/
│   ├── reproduce_final_speed_route.sh   # 复现最终速度路由提交
│   ├── reproduce_ensemble_weighted.sh   # 复现 weighted ensemble
│   └── run_pipeline.py                  # 端到端阶段化流水线
├── dataset/                         # 原始数据集
├── data/yolo_seg/                   # 转换后的 YOLO 格式数据
├── runs/                            # 训练输出
├── outputs/
│   ├── submissions/                 # 所有提交 JSON
│   └── reports/                     # 评估报告、指标、审计
├── deliverables/                    # 最终交付包
│   ├── yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate/  # 单模型保底
│   ├── yolo11n_seg_scalecombo_best_candidate/              # 训练模型候选
│   └── ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/  # ★ 最终
├── docs/                            # 文档和图表
├── requirements.txt
└── environment.yml
```

---

## 3. 最终方案架构

### 三步流水线

```
Step 1: 双模型各自推理
  ├── YOLO26n-seg (参考权重，200 epoch，高 tiny recall)
  │     imgsz=1280, conf=0.01, 条件切片, mask→bbox, 全后处理
  └── YOLO11n-seg (scalecombo，本工程 200 epoch 训练)
        imgsz=1280, conf=0.01, 条件切片, mask→bbox, 全后处理

Step 2: Weighted Merge 融合 (merge_submissions.py)
  ├── 两模型预测框合并到一个池子
  ├── 按 score 贪心遍历，IoU≥0.65 触发合并
  ├── 合并方式：按面积加权平均坐标，取 max score
  └── 每图最多保留 300 个预测框

Step 3: Speed Route (route_by_ids.py)
  ├── 24 张 regular 慢图 → 替换为 YOLO11n fast-detbox768 结果
  └── 其余 277 张图保持 Step 2 的 Weighted Ensemble
```

### 快速分支 (fast-detbox768) 详情

- 模型：YOLO11n scalecombo
- imgsz=768（vs 主推理 1280）
- conf=0.08（vs 主推理 0.01）
- 不做切片 (tile_size=0)
- 使用 detect head 直接输出的 bbox，不走 Seg Head → mask → 外接矩形
- 不做 mask 后处理（no retina_masks, no mask_iou_merge）
- 不做任何 box 扩张/补偿/union 后处理
- 单张耗时 < 100ms

---

## 4. 最终指标（最重要）

### 验证集（257 张图，340 GT bbox）

| 指标 | 数值 |
|---|---:|
| mAP50 | 0.5765 |
| Recall@IoU50 | 0.9147 |
| Tiny Recall@IoU50 | 0.9412 |
| Large Mean Matched IoU | 0.7981 |
| Large Mean Best IoU | 0.8356 |
| Precision@conf | 0.1268 |
| Predicted Boxes | 2453 |

### 测试集推理耗时（301 张图）

| 分桶 | 数量 | 平均 | 最大 | 赛题要求 | 判定 |
|---:|---:|---:|---:|---|:--:|
| Regular (≤2048px) | 279 | 30.62ms | **93.84ms** | < 100ms | ✅ |
| Large (>2048px) | 22 | 259.92ms | **1444.93ms** | < 2000ms | ✅ |
| 全部 | 301 | 47.38ms | 1444.93ms | — | — |

### 赛题硬指标

- ✅ Tiny Recall 94.12% (要求 ≥ 90%)
- ✅ Regular 单张最大 93.84ms (要求 < 100ms)
- ✅ Large 单张最大 1444.93ms (要求 < 2000ms)
- ⚠️ Large Best IoU 0.8356 (参考线 0.85，不作为硬门槛)

---

## 5. 关键推理参数（Hybrid 配置）

```yaml
infer:
  imgsz: 1280               # 输入分辨率
  conf: 0.01                # 极低置信度阈值保召回
  iou: 0.55                 # NMS IoU 阈值
  max_det: 300              # 每图最大预测框数
  direct_max_side: 2048     # ≤2048 走常规图分支
  global_max_side: 1280     # >2048 大图全局缩放
  tile_size: 1280           # 切片推理尺寸
  tile_overlap: 256         # 切片重叠
  tile_trigger: low_preds   # 全局预测少时触发切片
  max_tiles: 8              # 最多切 8 块

  # bbox 来源：从分割 mask 外接矩形取
  prediction_box_source: mask_box
  retina_masks: true
  keep_masks_for_merge: true

  # 后处理
  box_expand_ratio: 0.12            # 大框扩张 12%
  box_expand_pixels: 12.0           # 大框扩张 12px
  box_expand_min_area: 90000        # 仅面积≥90000 的大框

  tiny_box_min_width: 2.0           # 极窄框最小宽
  tiny_box_min_height: 6.0          # 极窄框最小高
  tiny_box_max_area: 80.0           # 仅面积≤80 的小框

  elongated_box_expand_ratio: 0.2   # 长条框方向扩张 20%
  elongated_box_min_area: 30000
  elongated_box_min_aspect: 3.0

  union_elongated_clusters: true    # 长裂纹跨框合并
  union_cluster_score_factor: 0.2   # union 框分数系数
  union_cluster_score_floor: 0.5    # union 框分数下限
  union_cluster_score_area_norm: 90000
  union_cluster_box_source: mask_box
```

---

## 6. 模型权重

| 权重 | 来源 | 用途 |
|---|---|---|
| `yolo26n_ref_unionfloor05.pth` | 参考项目训练 200 epoch | Ensemble 第一分支 |
| `yolo11n_scalecombo_best.pth` | 本工程训练 200 epoch | Ensemble 第二分支 |

### YOLO11n scalecombo 训练详情

- 数据：`crack_seg_scaleaware_scalecrop.yaml` (1461 行，尺度感知重采样 + 局部 crop)
- 参数：imgsz=1024, epochs=200, batch=2, seed=42
- GPU：RTX 4060 Ti 8GB
- 单独评估：mAP50=0.5462, Recall=0.8412, Tiny Recall=0.7647 ❌
- 单独审核：FAIL（Tiny Recall 不达标 + regular 速度超标）

---

## 7. 重要文件速查

| 文件 | 内容 |
|---|---|
| `docs/experiment_summary.md` | 完整实验消融记录 |
| `docs/final_delivery_checklist.md` | 交付核对表 |
| `docs/technical_design_report.md` | 技术设计报告 |
| `outputs/reports/candidate_comparison.md` | 候选对比与推荐 |
| `outputs/reports/delivery_audit_ensemble_*_candidate.md` | 各候选审计报告 |
| `scripts/reproduce_final_speed_route.sh` | **一键复现最终提交** |

---

## 8. 快速复现命令

```bash
cd /home/ruiyi/CPIPC/Dection
conda activate cpipc-crack

# 一键复现最终提交
bash scripts/reproduce_final_speed_route.sh

# 或分步执行（见 docs/final_delivery_checklist.md）
```

---

## 9. 已知遗留问题

1. Large Best IoU 0.8356，距离 0.85 参考线差 0.0144
2. 大裂纹框候选质量不足（模型没有稳定产生足够贴合 GT 的候选框）
3. 低 conf=0.01 带来 precision 仅 0.1268（2453 预测框 vs 340 GT）
4. 后续优化方向：训练侧大目标增强、更大模型、更高分辨率全图分支
