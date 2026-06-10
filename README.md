# 跨尺度芯片图像裂纹缺陷检测

本项目对应“赛题四：跨尺度芯片图像的裂纹缺陷智能检测算法设计”。方案采用 Ultralytics YOLO 目标检测路线，输出裂纹 `crack` 的 bbox，不输出 mask。官方提交说明允许目标检测模型不输出 mask，评测重点包含极小裂纹 Recall、极大裂纹 bbox IoU、全测试集 bbox mAP50 和平均推理耗时。

## 已检查的数据事实

- 数据目录：`dataset/trainval/images + trainval.json`，`dataset/test/images + test.json`。
- 训练图像 1285 张，测试图像 301 张，训练标注 1652 个 bbox，类别只有 `crack`。
- 全部图像为单通道灰度图；训练图像宽 45-7468、高 46-9267，测试图像宽 46-7445、高 47-9250。
- 极小目标按 `width<=5 or area<=50` 统计为 79 个；极大目标按 `area>=300*300` 统计为 83 个。
- 部分 bbox 贴边或有浮点级越界，转换时会 clip 到图像范围。

## 项目结构

```text
configs/yolo_crack.yaml    # 数据路径、训练参数、推理阈值、滑窗参数
src/common.py              # JSON/YAML、bbox、IoU、NMS、图像工具
src/data_analyze.py        # 数据集统计入口 main()
src/prepare_yolo.py        # 官方 JSON 转 YOLO 格式入口 main()
src/train_yolo.py          # YOLO 训练入口 main()
src/validate_yolo.py       # 本地验证和错误分析入口 main()
src/infer_submit.py        # 测试集推理并生成 results.json 入口 main()
src/check_submit.py        # 提交文件合法性检查入口 main()
requirements.txt           # 依赖说明
```

`dataset/`、`data/`、`outputs/`、`runs/`、权重文件和虚拟环境已加入 `.gitignore`，不会进入 Git。

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

## 环境安装

当前系统检测到 RTX 4060 Ti 8GB，但已有 PyTorch 是 CPU 版。建议创建虚拟环境并安装 CUDA 版 PyTorch，再安装本项目依赖。

```bash
cd /home/ruiyi/CPIPC/Dection
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 按实际 CUDA 版本选择官方命令；下面是 CUDA 12.1 示例
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

如果只运行数据统计和格式转换，现有环境已具备 `Pillow/PyYAML/sklearn/numpy`，不需要先安装 YOLO。

## 运行顺序

```bash
cd /home/ruiyi/CPIPC/Dection
source .venv/bin/activate  # 如已创建虚拟环境

python src/data_analyze.py --dataset dataset --out outputs/reports/data_stats.json
python src/prepare_yolo.py --dataset dataset --out data/yolo --val-ratio 0.2 --seed 42

python src/train_yolo.py --config configs/yolo_crack.yaml --model yolov8s.pt --imgsz 1024 --epochs 100 --batch 4
python src/validate_yolo.py --config configs/yolo_crack.yaml --weights runs/crack_yolo/train/weights/best.pt
python src/infer_submit.py --config configs/yolo_crack.yaml --weights runs/crack_yolo/train/weights/best.pt --split test
python src/check_submit.py --dataset dataset --submit outputs/submissions/results.json
```

8GB 显存默认使用 `batch=4,imgsz=1024`。若显存不足，先降为 `batch=2`；若在 16GB 或 4080 环境测速，可尝试 `imgsz=1280,batch=8`。

## 输出文件

- `outputs/reports/data_stats.json`：数据数量、尺度、类别、bbox 异常和难点样本统计。
- `data/yolo/crack.yaml`：Ultralytics 数据配置。
- `data/yolo/split_manifest.yaml`：训练/验证 ID 切分，便于复现实验。
- `runs/crack_yolo/train/weights/best.pt`：验证集最优权重。
- `outputs/reports/val_metrics.json`：`mAP50`、`precision_at_conf`、`recall_at_iou50`、`tiny_recall_at_iou50`、`large_mean_best_iou`、平均推理耗时。
- `outputs/reports/val_errors.csv`：验证集漏检样本，用于误差分析。
- `outputs/submissions/results.json`：最终提交文件。

## 调优方向

- 小目标：提高训练和推理分辨率、启用滑窗、降低 `conf` 并用验证集搜索阈值。
- 大目标：保留低分辨率整图分支，和切片结果一起 NMS，减少宏观裂纹被切碎。
- 跨尺度：比较 `tile_size=1024/1280/1536`、`tile_overlap=192/256/384`。
- 速度：训练完成后导出 ONNX/TensorRT，在官方 4080 环境复测 `inference_time_ms`。
