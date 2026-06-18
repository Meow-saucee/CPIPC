# CPIPC 赛题四 传输包说明

> 生成时间：2026-06-18 22:00 CST
> 目标机器：远程 Windows + RTX 4080

---

## 原主机信息

| 项目 | 值 |
|---|---|
| 项目路径 | `/home/ruiyi/CPIPC/Dection` |
| Git commit | `f629dd8979843a8d7371dcebd6e52936c45cf9b4` |
| Git commit 说明 | `Add Codex project handoff` |
| Python | 3.10.9 |
| Conda env | `cpipc-crack` |
| GPU | NVIDIA GeForce RTX 4060 Ti 8GB |
| 驱动 | 550.54.14 |
| PyTorch | 2.3.1+cu121 |
| torchvision | 0.18.1+cu121 |
| ultralytics | 8.4.63 |
| opencv-python | 4.13.0.92 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| PyYAML | 6.0.3 |

---

## 传输包内容

### 完整复制的目录

| 目录 | 大小 | 说明 |
|---|---|---|
| `dataset/` | 985M | 原始竞赛数据（trainval + test images + JSON） |
| `data/` | 224M | 预处理后的 YOLO-seg 格式数据 |
| `outputs/` | 410M | 所有提交 JSON、评估指标、审计报告 |
| `runs/` | 149M | 训练 checkpoint（含 best.pt / last.pt） |
| `experiments/` | 78M | 归档实验（含 TensorBoard events） |
| `deliverables/` | 367M | 最终交付包（含权重 .pt/.pth + results.json） |
| `*.pt` `*.pth` | ~27M | 根目录 YOLO 预训练权重 |

### 最终候选名称

```
ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate
```

### 最终提交文件

```
deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/results.json
```

---

## Git 仓库（需单独 clone）

这些文件在 GitHub，**不在本压缩包内**：

```
git clone git@github.com:Meow-saucee/CPIPC.git
```

Git 仓库包含：`README.md` `configs/` `src/` `scripts/` `docs/` `environment.yml` `requirements.txt` `.gitignore` `CODEX_HANDOFF.md`

---

## 期望落盘结构（Windows 4080 电脑）

```
D:\CPIPC\                          ← 项目根目录
├── dataset\                       ← 从传输包解压
│   ├── trainval\
│   │   ├── images\
│   │   └── trainval.json
│   └── test\
│       ├── images\
│       └── test.json
├── data\                          ← 从传输包解压
│   └── yolo_seg\
├── outputs\                       ← 从传输包解压
│   ├── submissions\
│   └── reports\
├── runs\                          ← 从传输包解压
│   └── crack_yolo_seg\
├── experiments\                   ← 从传输包解压
├── deliverables\                  ← 从传输包解压 ★
│   └── ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate\
│       ├── results.json           ← 最终提交
│       ├── weights\               ← .pt + .pth 权重
│       ├── reports\               ← val_metrics / speed
│       ├── source\                ← 源码副本
│       ├── docs\                  ← 文档副本
│       ├── configs\
│       ├── REPRODUCE.md
│       ├── README.md
│       └── manifest.json
├── yolo26n.pt                     ← 从传输包解压
├── yolov8s.pt                     ← 从传输包解压
│
├── .git\                          ← Git clone 产物
├── .gitignore                     ← Git clone 产物
├── README.md                      ← Git clone 产物
├── CODEX_HANDOFF.md               ← Git clone 产物
├── environment.yml                ← Git clone 产物
├── requirements.txt               ← Git clone 产物
├── configs\                       ← Git clone 产物
├── src\                           ← Git clone 产物
├── scripts\                       ← Git clone 产物
└── docs\                          ← Git clone 产物
```

**操作顺序**：
1. 先在 `D:\CPIPC\` 下 `git clone git@github.com:Meow-saucee/CPIPC.git`
2. 再把本压缩包解压覆盖到同一目录（Windows 资源管理器拖拽合并）

---

## 解压后验证命令（在 4080 上跑）

```bash
# 1. 环境检查
conda env list
nvidia-smi
python --version

# 2. 确认关键文件存在
ls deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/results.json
ls deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/weights/yolo26n_ref_unionfloor05.pth
ls deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/weights/yolo11n_scalecombo_best.pth
ls dataset/test/test.json

# 3. 装环境并验证
conda env create -f environment.yml
conda activate cpipc-crack
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 文件清单（传输完整性核验）

见同目录下的 `FILE_MANIFEST.txt`。
