# 跨尺度芯片图像裂纹缺陷检测

本项目对应“赛题四：跨尺度芯片图像的裂纹缺陷智能检测算法设计”。当前主推方案采用 Ultralytics YOLO-seg 实例分割路线，训练时利用 `bbox + segmentation` 标注，推理时融合模型 bbox 与 mask 外接框，最终按官方要求提交裂纹 `crack` 的 bbox JSON。检测 baseline 仍保留为速度优先和保底对照。

## 已检查的数据事实

- 数据目录：`dataset/trainval/images + trainval.json`，`dataset/test/images + test.json`。
- 训练图像 1285 张，测试图像 301 张，训练标注 1652 个 bbox，类别只有 `crack`。
- 全部图像为单通道灰度图；训练图像宽 45-7468、高 46-9267，测试图像宽 46-7445、高 47-9250。
- 极小目标按 `width<=5 or area<=50` 统计为 79 个；极大目标按 `area>=300*300` 统计为 83 个。
- 55 个 bbox 在裁剪到图像范围后会发生变化，多为贴边或浮点级越界，转换时会统一 clip。

## 项目结构

```text
configs/yolo_crack.yaml    # 数据路径、训练参数、推理阈值、滑窗参数
src/common.py              # JSON/YAML、bbox、IoU、NMS、图像工具
src/data_analyze.py        # 数据集统计入口 main()
src/prepare_yolo.py        # 官方 JSON 转 YOLO 格式入口 main()
src/train_yolo.py          # YOLO 训练入口 main()
src/validate_yolo.py       # 本地验证和错误分析入口 main()
src/infer_submit.py        # 测试集推理并生成 results.json 入口 main()
src/prepare_yolo_seg.py    # 官方 RLE mask 转 YOLO-seg 格式入口 main()
src/train_yolo_seg.py      # YOLO-seg 实例分割训练入口 main()
src/infer_submit_seg.py    # YOLO-seg 推理、mask 转 bbox 并生成提交入口 main()
src/diagnose_large_iou.py  # 大裂纹 matched/best/top-score 诊断入口 main()
src/check_submit.py        # 提交文件合法性检查入口 main()
src/visualize_predictions.py # GT/预测框可视化入口 main()
src/archive_experiment.py  # 归档已有 run，保存 ckpt/结果/TensorBoard 入口 main()
src/experiment_utils.py    # 实验命名、results.csv 汇总、TensorBoard 写入工具
docs/model_system_architecture.md # 输入输出、模型框架、跨尺度推理和关键参数图解
docs/model_framework_and_parameters.md # YOLO-seg 架构图、训练/推理产物和调参入口
docs/model_architecture_overview.md # 一页式模型输入输出、网络架构、推理后处理和改参入口
docs/assets/*.svg       # 可直接打开的系统流程图、模型架构图、推理后处理图
docs/experiment_summary.md # 已实际运行的模型、指标、耗时和推荐提交
docs/technical_design_report.md # 技术设计报告
docs/defense_slides_outline.md # 答辩演示提纲
docs/final_delivery_checklist.md # 最终交付核对表
requirements.txt           # 依赖说明
```

`dataset/`、`data/`、`outputs/`、`runs/`、`experiments/`、权重文件和虚拟环境已加入 `.gitignore`，不会进入 Git。

## 模型系统图

如果 Markdown 预览支持 Mermaid，可以直接阅读：

- [docs/model_system_architecture.md](/home/ruiyi/CPIPC/Dection/docs/model_system_architecture.md)
- [docs/model_framework_and_parameters.md](/home/ruiyi/CPIPC/Dection/docs/model_framework_and_parameters.md)
- [docs/model_architecture_overview.md](/home/ruiyi/CPIPC/Dection/docs/model_architecture_overview.md)

如果需要直接打开图片，查看：

- [docs/assets/system_pipeline.svg](docs/assets/system_pipeline.svg)：从数据准备、训练、推理到提交的整体系统。
- [docs/assets/yolo_seg_architecture.svg](docs/assets/yolo_seg_architecture.svg)：YOLO-seg Backbone、Neck、Detect Head、Seg Head 与输出关系。
- [docs/assets/inference_postprocess.svg](docs/assets/inference_postprocess.svg)：整图/全图缩放/切片推理与裂纹后处理流程。

## GitHub 回溯流程

本目录已可作为 Git 仓库使用。你在 GitHub 创建空仓库后，把远端地址替换到下面命令：

```bash
cd /home/ruiyi/CPIPC/Dection
git add .gitignore README.md requirements.txt configs src 背景.txt
git commit -m "Add YOLO crack detection pipeline"
git branch -M main
git remote add origin <REMOTE_URL>
git push -u origin main
git log --oneline -5
```

若 `remote origin already exists`，先执行 `git remote -v` 确认，必要时用 `git remote set-url origin <REMOTE_URL>`。

## 代码版本回溯操作

Git 可以支持三种常用回溯方式：查看历史、临时回到旧版本查看、彻底回退到旧版本。即使彻底回退，也可以用 `git reflog` 找回回退前的版本。

查看版本历史：

```bash
git log --oneline --decorate --graph --all
```

临时回到某个旧版本查看代码，不改变分支历史：

```bash
git switch --detach <commit_id>

# 查看完以后回到最新 main
git switch main
```

彻底回退当前分支到某个旧版本，会让工作区和分支指针都回到该版本：

```bash
git status
git reset --hard <commit_id>
```

彻底回退后如果发现回错了，可以用 `reflog` 找到回退前的提交，再恢复：

```bash
git reflog
git reset --hard <commit_id_before_reset>
```

如果已经把错误回退推送到了 GitHub，需要强制同步远端。执行前务必确认 `git log --oneline -5` 显示的是想要保留的版本：

```bash
git push --force-with-lease origin main
```

更保守的回退方式是 `git revert`，它不会改写历史，而是新增一个“撤销某次提交”的提交；适合多人协作：

```bash
git revert <commit_id>
git push origin main
```

当前项目已经有两个可回溯提交：

```text
1091a80 Document reproduced dataset stats
d417f85 Add YOLO crack detection pipeline
```

## Conda 环境安装

当前项目已在 Conda 环境 `cpipc-crack` 下验证通过，GPU 为 RTX 4060 Ti 8GB，核心版本如下：

```text
python 3.10.20
torch 2.3.1+cu121
torchvision 0.18.1+cu121
ultralytics 8.4.63
cuda_available True
```

已验证的手工安装命令：

```bash
cd /home/ruiyi/CPIPC/Dection
conda create -y -n cpipc-crack python=3.10 pip
conda activate cpipc-crack
pip install --upgrade pip
pip install --no-cache-dir torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

也可以尝试使用 `environment.yml` 一次性创建环境：

```bash
conda env create -f environment.yml
conda activate cpipc-crack
```

如果只运行数据统计和格式转换，现有环境已具备 `Pillow/PyYAML/sklearn/numpy`，不需要先安装 YOLO。

## 运行顺序

```bash
cd /home/ruiyi/CPIPC/Dection
conda activate cpipc-crack

python src/data_analyze.py --dataset dataset --out outputs/reports/data_stats.json
python src/prepare_yolo.py --dataset dataset --out data/yolo --val-ratio 0.2 --seed 42

python src/train_yolo.py --config configs/yolo_crack.yaml --model yolov8s.pt --imgsz 1024 --epochs 100 --batch 4
python src/validate_yolo.py --config configs/yolo_crack.yaml --weights runs/crack_yolo/train/weights/best.pt
python src/infer_submit.py --config configs/yolo_crack.yaml --weights runs/crack_yolo/train/weights/best.pt --split test
python src/check_submit.py --dataset dataset --submit outputs/submissions/results.json
```

8GB 显存默认使用 `batch=4,imgsz=1024`。若显存不足，先降为 `batch=2`；若在 16GB 或 4080 环境测速，可尝试 `imgsz=1280,batch=8`。

## YOLO-seg 高分路线

训练集同时提供 `segmentation` 和 `bbox`。如果希望更充分利用裂纹像素级标注，可以使用实例分割路线训练，再把 mask 外接矩形转换为官方 `predict_bboxes` 提交。

```bash
cd /home/ruiyi/CPIPC/Dection
conda activate cpipc-crack

python src/prepare_yolo_seg.py --config configs/yolo_seg_crack.yaml
python src/train_yolo_seg.py --config configs/yolo_seg_crack.yaml --model yolo11n-seg.pt --imgsz 1024 --epochs 200 --batch 2
python src/infer_submit_seg.py --config configs/yolo_seg_crack.yaml --weights runs/crack_yolo_seg/<experiment_name>/weights/best.pt --split test
python src/check_submit.py --dataset dataset --submit outputs/submissions/results_seg.json
```

`configs/yolo_seg_crack.yaml` 中可调整 `segmentation.epsilon_ratio`、`segmentation.max_points`、`train.mask_ratio`、`infer.tile_size`、`infer.tile_overlap`、`infer.conf`、`infer.box_iou_merge` 和 `infer.mask_iou_merge`。小裂纹召回不足时优先降低 `infer.conf`、提高 `imgsz` 或使用更密的切片；极大裂纹定位不稳定时保留 `include_global_for_large: true`。

尺度感知重采样训练可作为下一轮提升 tiny recall 和 large IoU 的实验入口。该方法不复制图片和标签，只生成重复采样的 `train_scaleaware.txt` 与新的 YOLO data yaml，验证集保持不变：

```bash
python src/build_scale_aware_train_list.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --out-suffix scaleaware \
  --tiny-repeat 3 \
  --large-repeat 3 \
  --huge-repeat 2 \
  --tiny-large-repeat 4 \
  --max-repeat 4

python src/train_yolo_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --data-yaml data/yolo_seg/crack_seg_scaleaware.yaml \
  --model yolo11n-seg.pt \
  --imgsz 1024 \
  --epochs 200 \
  --batch 2 \
  --tag seg-scaleaware
```

当前已生成的 scale-aware 清单统计：基础训练图 1028 张，重复后训练列表 1274 行；其中 tiny 图 51 张、large 图 64 张、超大图 43 张被提高采样频率。注意：这只是已生成训练入口，尚未完成长训验证，不能作为已提升指标报告。

局部尺度 crop 训练可进一步增强 tiny/large 样本。它会只从 train split 中裁剪局部图，不改变验证集；tiny 裂纹在 crop 中相对变大，large 裂纹会提供局部细节视角：

```bash
python src/build_scale_crop_dataset.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --out-suffix scalecrop \
  --crop-size 1024 \
  --context 2.5 \
  --tiny-repeat 2 \
  --large-repeat 1 \
  --max-crops-per-image 4

python src/train_yolo_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --data-yaml data/yolo_seg/crack_seg_scalecrop.yaml \
  --model yolo11n-seg.pt \
  --imgsz 1024 \
  --epochs 200 \
  --batch 2 \
  --tag seg-scalecrop
```

当前已生成 scale-crop 数据：原始 train 1028 行，新增 crop 187 张，合并训练列表 1215 行；其中 tiny crop 120 张、large crop 67 张，标签文件均非空。该训练集尚未完成长训验证。

推荐的下一轮长训候选是“整图尺度重采样 + 局部 crop”的组合训练集：

```bash
python src/build_combined_yolo_data.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --out-suffix scaleaware_scalecrop \
  --train-lists data/yolo_seg/train_scaleaware.txt data/yolo_seg/train_scalecrop_only.txt

python src/train_yolo_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --data-yaml data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml \
  --model yolo11n-seg.pt \
  --imgsz 1024 \
  --epochs 200 \
  --batch 2 \
  --tag seg-scaleaware-scalecrop
```

当前组合清单已生成并校验：训练列表 1461 行，唯一图像/crop 1215 个，缺失图片 0，缺失标签 0，空标签 0。

长训前建议再运行 YOLO-seg 数据预检：

```bash
python src/check_yolo_seg_data.py \
  --data-yaml data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml \
  --out outputs/reports/check_crack_seg_scaleaware_scalecrop.json
```

当前预检结果：`ok=True`，train 1461 行，train unique 1215，重复 246 行用于过采样，val 257 行，test 301 行，failures 为空。

组合数据 smoke train 已跑通，用于确认训练链路可用：

```bash
python src/train_yolo_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --data-yaml data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml \
  --model /home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/yolo11n-seg.pt \
  --imgsz 640 \
  --epochs 1 \
  --batch 1 \
  --device 0 \
  --name yolo11n_seg_scalecombo_smoke \
  --tag seg-scalecombo-smoke \
  --no-archive
```

产物：

```text
runs/crack_yolo_seg/yolo11n_seg_scalecombo_smoke/weights/best.pt
runs/crack_yolo_seg/yolo11n_seg_scalecombo_smoke/weights/last.pt
runs/crack_yolo_seg/yolo11n_seg_scalecombo_smoke/args.yaml
runs/crack_yolo_seg/yolo11n_seg_scalecombo_smoke/results.csv
runs/crack_yolo_seg/yolo11n_seg_scalecombo_smoke/results.png
```

该 smoke run 成功扫描 train 1461 张、val 257 张，0 corrupt，并完成 1 epoch 训练和验证。`mAP50(B)=0.0089` 仅说明 1 epoch smoke 模型很弱，不作为正式指标。

也可以使用流水线脚本统一调度阶段：

```bash
# 只打印 prepare/check/smoke 命令，不执行
python scripts/run_pipeline.py --stages prepare check smoke --dry-run

# 执行组合训练数据预检
python scripts/run_pipeline.py --stages check

# 后续正式长训时显式执行 train 阶段
python scripts/run_pipeline.py \
  --stages train \
  --train-data-yaml data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml \
  --pretrained-seg /home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/yolo11n-seg.pt \
  --train-imgsz 1024 \
  --epochs 200 \
  --batch 2 \
  --device 0
```

也可以使用当前固定的正式训练脚本：

```bash
conda activate cpipc-crack
bash scripts/train_scalecombo_200e.sh
```

训练过程中查看进度、best epoch 和预计剩余时间：

```bash
RUN=runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop
python scripts/monitor_training.py --run-dir "$RUN" --epochs 200
nvidia-smi
```

训练完成后，一键完成验证、测试集推理、测速、打包和审计：

```bash
bash scripts/eval_package_scalecombo.sh \
  runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop \
  yolo11n_seg_scalecombo_best_candidate
```

流水线脚本默认不会自动跑 200 epoch；必须显式指定 `--stages train` 才会启动正式训练。

当前已跑通的候选模型和指标见：

```text
docs/experiment_summary.md
```

当前推荐提交候选：

```text
候选：ensemble_weighted_route_regular_gt100_fastdetbox768_warm
权重：deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/weights/yolo26n_ref_unionfloor05.pth
权重：deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/weights/yolo11n_scalecombo_best.pth
配置：configs/yolo_seg_crack_hybrid.yaml
提交：outputs/submissions/results_ensemble_weighted_route_regular_gt100_fastdetbox768_warm.json
交付包：deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate
验证：mAP50=0.5765, Recall=0.9147, Tiny Recall=0.9412, Large Matched IoU=0.7981, Large Best IoU=0.8356
测试耗时：avg=47.38ms, regular max=93.84ms, large max=1444.93ms
```

说明：`ensemble_y26_y11_weighted` 是验证 mAP50/召回优先基线；当前最终提交在该基线基础上，对 24 张 regular 历史慢图路由到 `yolo11n fast detbox768` 分支，使 regular 单张耗时降到 100ms 内。`ensemble_y26_y11_w075_calibrated_demote` 保留为大裂纹定位参考指标更高的备选。

复现当前 ensemble 提交：

```bash
bash scripts/reproduce_final_speed_route.sh
```

模型输入输出、YOLO Backbone/Neck/Head、实例分割 mask 转 bbox、跨尺度整图/切片推理流程和关键参数位置见：

```text
docs/model_io_architecture_cheatsheet.md
docs/model_system_architecture.md
docs/model_framework_and_parameters.md
docs/model_architecture_overview.md
docs/speed_route_strategy.md
```

技术报告、答辩提纲和交付核对表：

```text
docs/technical_design_report.md
docs/defense_slides_outline.md
docs/final_delivery_checklist.md
```

快速测速/提交配置默认禁用切片，适合先满足推理耗时约束：

```bash
python src/infer_submit_seg.py \
  --config configs/yolo_seg_crack_fast.yaml \
  --weights runs/crack_yolo_seg/<experiment_name>/weights/best.pt \
  --split test \
  --out outputs/submissions/results_seg_fast.json
```

也可以直接用命令行覆盖推理参数做消融：

```bash
python src/infer_submit_seg.py \
  --config configs/yolo_seg_crack.yaml \
  --weights runs/crack_yolo_seg/<experiment_name>/weights/best.pt \
  --tile-size 0 \
  --direct-max-side 2048 \
  --global-max-side 1280 \
  --conf 0.08 \
  --out outputs/submissions/results_seg_fast.json
```

Hybrid 配置会先跑快速全图分支，再对低预测数的大图补少量切片，目标是在速度约束内提高召回：

```bash
python src/infer_submit_seg.py \
  --config configs/yolo_seg_crack_hybrid.yaml \
  --weights runs/crack_yolo_seg/<experiment_name>/weights/best.pt \
  --split test \
  --out outputs/submissions/results_seg_hybrid.json
```

## 可视化

训练集 GT 可视化：

```bash
python src/visualize_predictions.py --dataset dataset --split trainval --limit 20
```

测试集提交结果可视化：

```bash
python src/visualize_predictions.py --dataset dataset --split test --submit outputs/submissions/results.json --limit 20
python src/visualize_predictions.py --dataset dataset --split test --submit outputs/submissions/results_seg.json --limit 20
```

推理耗时汇总：

```bash
python src/summarize_submission_time.py --submit outputs/submissions/results.json
python src/summarize_submission_time.py \
  --submit outputs/submissions/results_seg.json \
  --out-json outputs/reports/inference_time_summary_seg.json \
  --out-csv outputs/reports/inference_time_per_image_seg.csv
```

按最终提交 JSON 口径评估验证集预测：

```bash
python src/infer_submit_seg.py \
  --config configs/yolo_seg_crack_fast.yaml \
  --weights runs/crack_yolo_seg/<experiment_name>/weights/best.pt \
  --split val \
  --out outputs/submissions/val_pred_seg_fast.json

python src/eval_submission.py \
  --config configs/yolo_seg_crack_fast.yaml \
  --submit outputs/submissions/val_pred_seg_fast.json \
  --split val \
  --out outputs/reports/submission_metrics_seg_fast_val.json
```

## 交付打包

训练和推理完成后，可以把最佳权重、提交结果、配置、指标和复现说明整理到 `deliverables/<name>/`。脚本会额外生成 `.pth` 命名的权重副本，便于满足赛题交付格式。

```bash
python src/package_delivery.py \
  --name yolo11n_seg_final \
  --weights runs/crack_yolo_seg/<experiment_name>/weights/best.pt \
  --submission outputs/submissions/results_seg.json \
  --config configs/yolo_seg_crack.yaml \
  --metrics outputs/reports/val_metrics_seg.json \
  --errors outputs/reports/val_errors.csv
```

提交前建议再运行交付验收审计。该脚本会同时检查交付包文件、`.pth` 权重、提交 JSON 合法性、验证集召回与定位参考指标、常规图/超大图推理耗时和报告完整性：

```bash
python src/audit_delivery.py \
  --delivery deliverables/yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate \
  --dataset dataset \
  --out-json outputs/reports/delivery_audit_unionfloor05_strict.json \
  --out-md outputs/reports/delivery_audit_unionfloor05_strict.md
```

也可以通过阶段化流水线运行：

```bash
python scripts/run_pipeline.py --stages audit
```

当前最终候选包 `ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate` 的交付审计结论为 `PASS`：Tiny Recall、提交格式、常规图单张耗时和超大图耗时均通过；Large IoU 当前作为定位参考指标记录，不作为交付硬门槛。

## 实验命名、归档和 TensorBoard

默认配置会按模型、数据集、关键训练参数生成实验名：

```text
{model}_{dataset}_img{imgsz}_ep{epochs}_bs{batch}_seed{seed}_{tag}
```

例如本次正式基线实验名为：

```text
yolov8s_cpipc-chip-crack_img1024_ep100_bs4_seed42_baseline
```

后续训练默认会把 Ultralytics run 写到 `runs/crack_yolo/<experiment_name>`，训练完成后自动归档到 `experiments/<experiment_name>`。归档目录结构：

```text
experiments/<experiment_name>/
  checkpoints/        # 标准命名 best/last ckpt
  reports/            # results.csv、args.yaml、自定义验证、错误分析、提交文件
  plots/              # Ultralytics 曲线、混淆矩阵、batch 可视化
  tensorboard/        # 从 results.csv 生成的 events.out.tfevents.*
  experiment.json     # last_epoch、best_epoch、best_metrics、路径和命令
  experiment.yaml
```

手动归档已有 run：

```bash
python src/archive_experiment.py \
  --run-dir runs/crack_yolo/train \
  --config configs/yolo_crack.yaml \
  --model yolov8s.pt \
  --dataset-name cpipc-chip-crack \
  --imgsz 1024 \
  --epochs 100 \
  --batch 4 \
  --seed 42 \
  --tag baseline \
  --metrics outputs/reports/baseline_val_metrics.json \
  --errors outputs/reports/val_errors.csv \
  --submission outputs/submissions/results.json
```

查看 TensorBoard：

```bash
tensorboard --logdir experiments --host 0.0.0.0 --port 6006
```

若只想训练不自动归档，可加 `--no-archive`。若要强制使用自定义 run 名，可加 `--name your_exp_name`，但建议仍保留模型、数据集和关键参数。

## 输出文件

- `outputs/reports/data_stats.json`：数据数量、尺度、类别、bbox 异常和难点样本统计。
- `data/yolo/crack.yaml`：Ultralytics 数据配置。
- `data/yolo/split_manifest.yaml`：训练/验证 ID 切分，便于复现实验。
- `runs/crack_yolo/train/weights/best.pt`：验证集最优权重。
- `outputs/reports/val_metrics.json`：`mAP50`、`precision_at_conf`、`recall_at_iou50`、`tiny_recall_at_iou50`、`large_mean_best_iou`、平均推理耗时。
- `outputs/reports/val_errors.csv`：验证集漏检样本，用于误差分析。
- `outputs/submissions/results.json`：最终提交文件。
- `experiments/<experiment_name>/experiment.json`：规范实验索引，包含 `last_epoch`、`best_epoch`、best/last ckpt 路径、训练参数和 TensorBoard 路径。
- `outputs/reports/delivery_audit_*.json/.md`：交付包完整性、提交合法性、关键指标和速度目标审计报告。

## 调优方向

- 小目标：提高训练和推理分辨率、启用滑窗、降低 `conf` 并用验证集搜索阈值。
- 大目标：保留低分辨率整图分支，和切片结果一起 NMS，减少宏观裂纹被切碎。
- 跨尺度：比较 `tile_size=1024/1280/1536`、`tile_overlap=192/256/384`。
- 速度：训练完成后导出 ONNX/TensorRT，在官方 4080 环境复测 `inference_time_ms`。
