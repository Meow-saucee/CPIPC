# Codex Project Handoff

This file is a portable handoff summary for opening this project in another Codex session.
It is not the full chat transcript. It captures the current project state, key decisions,
important files, reproducible commands, and the prompt to give the next Codex instance.

## 1. Repository And Current State

- Project path on the original machine: `/home/ruiyi/CPIPC/Dection`
- Git remote: `git@github.com:Meow-saucee/CPIPC.git`
- Base commit before this handoff file was added: `6db9c31`
- Main task: CPIPC problem 4, cross-scale chip image crack defect detection.
- Final method: YOLO instance segmentation plus bbox-level ensemble and speed routing.
- Final submission package:
  - `deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/`
- Final submission file:
  - `deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/results.json`
- Dataset is intentionally not tracked by Git. Expected local dataset path:
  - `dataset/trainval/images`
  - `dataset/trainval/trainval.json`
  - `dataset/test/images`
  - `dataset/test/test.json`

## 2. What Has Been Completed

- Dataset analysis and conversion scripts were implemented.
- YOLO detection baseline and YOLO segmentation pipeline were implemented.
- Experiment naming, checkpoint archival, TensorBoard export, and reproducibility docs were added.
- A 200 epoch `yolo11n-seg` scale-aware training run exists locally.
- Multiple validation and post-processing candidates were evaluated.
- Final candidate is a routed ensemble:
  - quality source: `ensemble_y26_y11_weighted`
  - speed route: regular slow images routed to `yolo11n fast detbox768`
- Final delivery audit passed.
- Submission format check passed for 301 test images.
- SVG XML bug in `docs/assets/inference_postprocess.svg` was fixed and pushed.

## 3. Current Best Candidate Metrics

Source:

- `deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/reports/val_metrics.json`
- `deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/reports/inference_time_summary.json`
- `deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/reports/speed_buckets.json`

Validation metrics:

- images: `257`
- ground truth boxes: `340`
- predicted boxes: `2453`
- mAP50: `0.5765244981471731`
- Recall@IoU50: `0.9147058823529411`
- Tiny Recall@IoU50: `0.9411764705882353`
- Large Matched IoU: `0.7981227040353821`
- Large Best IoU: `0.8355658219956857`

Test speed summary:

- test images: `301`
- average inference time: `47.38242192691029 ms`
- max inference time: `1444.93 ms`
- regular image max time after routing: `93.838 ms`
- delivery audit status: `PASS`

Important note:

- Large IoU is kept as a diagnostic metric only. The user explicitly said it is no longer necessary to optimize large-crack IoU to `0.85`.

## 4. Main Training Run

Formal local training run:

```text
runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop/
```

Key files:

```text
args.yaml
results.csv
weights/best.pt
weights/last.pt
results.png
BoxPR_curve.png
BoxF1_curve.png
BoxP_curve.png
BoxR_curve.png
MaskPR_curve.png
MaskF1_curve.png
MaskP_curve.png
MaskR_curve.png
confusion_matrix.png
confusion_matrix_normalized.png
labels.jpg
val_batch0_pred.jpg
val_batch1_pred.jpg
val_batch2_pred.jpg
```

Known training summary from `results.csv`:

- total epochs: `200`
- last epoch: `200`
- best epoch by `metrics/mAP50(B)`: `199`
- best `metrics/mAP50(B)`: `0.62668`
- epoch 199 Box P/R: `0.66188 / 0.59798`
- epoch 199 Mask P/R: `0.60295 / 0.52161`

Archived experiment path:

```text
experiments/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop/
```

TensorBoard:

```bash
tensorboard --logdir experiments --host 0.0.0.0 --port 6006
```

## 5. Files To Read First

For a new Codex session, read these files first:

```text
README.md
docs/model_quick_index.md
docs/model_system_architecture.md
docs/model_architecture_overview.md
docs/model_framework_and_parameters.md
docs/parameter_map.md
docs/experiment_summary.md
docs/reproduce_final_candidate.md
docs/final_delivery_checklist.md
docs/speed_route_strategy.md
```

Architecture SVGs:

```text
docs/assets/system_pipeline.svg
docs/assets/yolo_seg_architecture.svg
docs/assets/inference_postprocess.svg
```

Main config:

```text
configs/yolo_seg_crack_hybrid.yaml
```

Main source entrypoints:

```text
src/data_analyze.py
src/prepare_yolo.py
src/prepare_yolo_seg.py
src/train_yolo.py
src/train_yolo_seg.py
src/infer_submit.py
src/infer_submit_seg.py
src/eval_submission.py
src/check_submit.py
src/audit_delivery.py
src/package_delivery.py
src/archive_experiment.py
```

Main scripts:

```text
scripts/train_scalecombo_200e.sh
scripts/eval_package_scalecombo.sh
scripts/reproduce_final_speed_route.sh
scripts/monitor_training.py
scripts/wait_eval_scalecombo.py
```

## 6. Environment

Preferred environment files:

```text
environment.yml
requirements.txt
```

Typical setup on a new machine:

```bash
git clone git@github.com:Meow-saucee/CPIPC.git
cd CPIPC
conda env create -f environment.yml
conda activate cpipc-crack
```

If `environment.yml` fails because CUDA/PyTorch wheels differ by machine, install PyTorch
for the target CUDA version first, then install project dependencies from `requirements.txt`.

The dataset must be copied separately into:

```text
dataset/
```

The large generated folders may not be fully tracked by Git depending on `.gitignore`:

```text
dataset/
runs/
outputs/
deliverables/
experiments/
```

If these are missing on the new machine, copy them from the original machine or regenerate them.

## 7. Reproduce And Check Commands

Data check:

```bash
python src/data_analyze.py --dataset dataset --out outputs/reports/data_stats.json
python src/check_yolo_seg_data.py --data data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml --out outputs/reports/check_crack_seg_scaleaware_scalecrop_recheck.json
```

Train scale-aware YOLO-seg:

```bash
bash scripts/train_scalecombo_200e.sh
```

Monitor training:

```bash
RUN=runs/crack_yolo_seg/yolo11n-seg_cpipc-chip-crack-seg_img1024_ep200_bs2_seed42_seg-scaleaware-scalecrop
python scripts/monitor_training.py --run-dir "$RUN" --epochs 200
```

Evaluate and package a trained run:

```bash
bash scripts/eval_package_scalecombo.sh "$RUN" yolo11n_seg_scalecombo_best_candidate
```

Reproduce final speed-route candidate:

```bash
bash scripts/reproduce_final_speed_route.sh
```

Check final submission:

```bash
python src/check_submit.py \
  --dataset dataset \
  --submit deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate/results.json
```

Audit final delivery:

```bash
python src/audit_delivery.py \
  --delivery deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate \
  --dataset dataset \
  --benchmark-speed outputs/reports/benchmark_speed_ensemble_weighted_route_regular_gt100_fastdetbox768_warm.json \
  --out-json outputs/reports/delivery_audit_ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate.json \
  --out-md outputs/reports/delivery_audit_ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate.md
```

## 8. Git Version Rollback Notes

The user wants version history that supports:

- viewing old code versions
- temporarily checking out an older version
- permanently reverting to an older version
- undoing a revert if needed

Useful commands:

```bash
git log --oneline --decorate --graph --all
git show <commit>
git diff <old_commit> <new_commit>
```

Temporarily inspect an old version:

```bash
git switch --detach <commit>
```

Return to main:

```bash
git switch main
```

Create a safe revert commit:

```bash
git revert <commit>
```

Undo that revert:

```bash
git revert <revert_commit>
```

Avoid `git reset --hard` unless the user explicitly asks for destructive history changes.

## 9. Current Open Or Pending Items

- The user reported that VS Code cannot render Mermaid blocks in Markdown.
- Existing SVGs cover three main diagrams:
  - `docs/assets/system_pipeline.svg`
  - `docs/assets/yolo_seg_architecture.svg`
  - `docs/assets/inference_postprocess.svg`
- Some docs still contain Mermaid code blocks without inserted static SVG equivalents:
  - `docs/model_system_architecture.md`
  - `docs/model_architecture_overview.md`
  - `docs/model_io_architecture_cheatsheet.md`
  - `docs/model_framework_and_parameters.md`
  - `docs/technical_design_report.md`
  - `docs/defense_slides_outline.md`
- Environment has Graphviz `dot`, but no `node`, `npm`, `npx`, or `mmdc` detected in the previous check.
- If continuing that task, generate static SVGs from the Mermaid diagrams or draw equivalent Graphviz/manual SVG diagrams and insert image links above the Mermaid blocks.

## 10. Prompt For The Next Codex Session

Paste this into Codex on the new computer:

```text
请先阅读 CODEX_HANDOFF.md，然后阅读 README.md、docs/model_quick_index.md、
docs/model_system_architecture.md、docs/model_architecture_overview.md、
docs/model_framework_and_parameters.md、docs/parameter_map.md 和
docs/experiment_summary.md。

这是 CPIPC 赛题四“跨尺度芯片图像的裂纹缺陷智能检测算法设计”项目。
当前最终候选是 deliverables/ensemble_weighted_route_regular_gt100_fastdetbox768_warm_candidate。
不要编造指标；如果本机缺少 dataset/runs/outputs/deliverables/experiments，请先明确缺失内容。

请先检查 git status、数据集路径、environment.yml、最终交付包和结果文件。
之后根据我的新请求继续工作。每完成一个子任务后，主动向我汇报进度再继续。
```
