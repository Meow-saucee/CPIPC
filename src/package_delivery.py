from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def copy_required(src: Path, dst: Path) -> str:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def copy_optional(src: Path | None, dst: Path) -> str | None:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def copy_tree_files(
    src_dir: Path,
    dst_dir: Path,
    patterns: tuple[str, ...] = ("*.md",),
    recursive: bool = False,
) -> list[str]:
    copied: list[str] = []
    if not src_dir.exists():
        return copied
    for pattern in patterns:
        iterator = src_dir.rglob(pattern) if recursive else src_dir.glob(pattern)
        for src in iterator:
            if src.is_file():
                if "__pycache__" in src.parts:
                    continue
                rel = src.relative_to(src_dir) if recursive else Path(src.name)
                dst = dst_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied.append(str(dst))
    return copied


def copy_source_tree(src_dir: Path, dst_dir: Path) -> list[str]:
    copied: list[str] = []
    if not src_dir.exists():
        return copied
    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue
        if "__pycache__" in src.parts:
            continue
        if src.suffix in {".pyc", ".pt", ".pth"}:
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def write_reproduce_md(out_dir: Path, manifest: dict) -> None:
    config_name = Path(manifest["files"]["config"]).name
    weights_name = Path(manifest["files"]["weights_pth"]).name
    extra_weights = manifest["files"].get("extra_weights", [])
    is_ensemble = bool(extra_weights)
    delivery_name = str(manifest.get("name", ""))
    if "route_regular_gt100_fastdetbox768_warm" in delivery_name:
        ensemble_script = "scripts/reproduce_final_speed_route.sh"
    elif "w075_calibrated" in delivery_name:
        ensemble_script = "scripts/reproduce_ensemble_w075_calibrated.sh"
    else:
        ensemble_script = "scripts/reproduce_ensemble_weighted.sh"
    lines = [
        "# 复现实验说明",
        "",
        "本说明用于从交付包中的权重、配置和脚本复现最终测试集提交。",
        "",
        "## 交付包内容",
        "",
        "- `source/`：源码、配置和环境文件副本。",
        "- `weights/`：`.pt` 原始权重和 `.pth` 比赛交付权重副本。",
        "- `results.json`：测试集提交结果。",
        "- `reports/`：验证指标、错误分析和耗时报告。",
        "- `docs/`：技术报告、实验总结、架构图解和交付核对表。",
        "",
        "## 环境",
        "",
        "```bash",
        "cd /home/ruiyi/CPIPC/Dection",
        "conda env create -f environment.yml",
        "conda activate cpipc-crack",
        "```",
        "",
        "## 数据准备",
        "",
        "如果只使用交付包内源码，可先回到工程根目录并同步源码副本：",
        "",
        "```bash",
        "# 可选：交付包 source/ 目录中已包含 src/configs/requirements/environment 等文件",
        "```",
        "",
        "检测路线：",
        "",
        "```bash",
        "python src/prepare_yolo.py --dataset dataset --out data/yolo --val-ratio 0.2 --seed 42",
        "```",
        "",
        "分割路线：",
        "",
        "```bash",
        f"python src/prepare_yolo_seg.py --config configs/{config_name}",
        "```",
        "",
        "## 推理复现",
        "",
        "在完整工程目录下，可直接使用交付包中的 `.pth` 权重副本进行推理。Ultralytics 权重内容未改变，仅扩展名按比赛交付要求复制为 `.pth`。",
        "",
    ]
    if is_ensemble:
        lines.extend(
            [
                "### Ensemble 融合复现",
                "",
                "本交付包为双模型 bbox 融合候选。复现流程是先分别运行 yolo26n 与 yolo11n，再执行 bbox 融合与必要的后处理校准。",
                "",
                "```bash",
                f"bash {ensemble_script}",
                "```",
                "",
                "也可以手动指定输出路径：",
                "",
                "```bash",
                "OUT=outputs/submissions/reproduced_ensemble_results.json \\",
                f"bash {ensemble_script}",
                "```",
                "",
                "交付包中的 ensemble 权重：",
                "",
            ]
        )
        for item in extra_weights:
            lines.append(f"- `{item}`")
        lines.extend([""])
    lines.extend(
        [
            "### 单模型推理复现",
            "",
            "如果只想验证主权重能被 Ultralytics 加载，可运行：",
            "",
            "```bash",
            f"python src/infer_submit_seg.py --config configs/{config_name} \\",
            f"  --weights {out_dir / 'weights' / weights_name} \\",
            "  --split test \\",
            "  --out outputs/submissions/reproduced_results.json",
            "",
            "python src/check_submit.py \\",
            "  --dataset dataset \\",
            "  --submit outputs/submissions/reproduced_results.json",
            "```",
            "",
            "验证集评估：",
            "",
            "```bash",
            f"python src/infer_submit_seg.py --config configs/{config_name} \\",
            f"  --weights {out_dir / 'weights' / weights_name} \\",
            "  --split val \\",
            "  --out outputs/submissions/reproduced_val_results.json",
            "",
            f"python src/eval_submission.py --config configs/{config_name} \\",
            "  --submit outputs/submissions/reproduced_val_results.json \\",
            "  --split val \\",
            "  --out outputs/reports/reproduced_val_metrics.json",
            "```",
            "",
            "## 本交付包文件",
            "",
            "```json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    (out_dir / "REPRODUCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package model, submission, metrics and reproduction docs for delivery.")
    parser.add_argument("--name", required=True, help="Delivery name, e.g. yolo11n_seg_final.")
    parser.add_argument("--weights", required=True, help="Best Ultralytics .pt weights.")
    parser.add_argument("--extra-weight", action="append", default=[], help="Additional ensemble weight in name=path format.")
    parser.add_argument("--submission", required=True, help="Final results JSON.")
    parser.add_argument("--config", required=True, help="Main config used for inference/training.")
    parser.add_argument("--metrics", default=None, help="Validation metrics JSON.")
    parser.add_argument("--errors", default=None, help="Error analysis CSV.")
    parser.add_argument("--time-summary", default=None, help="Inference time summary JSON.")
    parser.add_argument("--speed-buckets", default=None, help="Inference speed bucket JSON.")
    parser.add_argument("--report", default=None, help="Technical report file if already prepared.")
    parser.add_argument("--copy-docs", action="store_true", help="Copy all markdown docs into the delivery package.")
    parser.add_argument("--copy-source", action="store_true", help="Copy source code, configs and env files into source/.")
    parser.add_argument("--out-root", default="deliverables")
    args = parser.parse_args()

    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_src = Path(args.weights)
    stem = Path(args.name).stem
    manifest = {
        "name": args.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": {},
    }
    manifest["files"]["weights_pt"] = copy_required(weights_src, out_dir / "weights" / f"{stem}.pt")
    # The contest asks for .pth. Ultralytics checkpoints are torch checkpoints, so
    # this creates a .pth-named copy while preserving checkpoint contents.
    manifest["files"]["weights_pth"] = copy_required(weights_src, out_dir / "weights" / f"{stem}.pth")
    extra_weights: list[str] = []
    for spec in args.extra_weight:
        if "=" not in spec:
            raise ValueError("--extra-weight must use name=path format")
        alias, src_text = spec.split("=", 1)
        alias = alias.strip()
        if not alias:
            raise ValueError("--extra-weight alias cannot be empty")
        src = Path(src_text)
        extra_weights.append(copy_required(src, out_dir / "weights" / f"{alias}.pt"))
        extra_weights.append(copy_required(src, out_dir / "weights" / f"{alias}.pth"))
    if extra_weights:
        manifest["files"]["extra_weights"] = extra_weights
    manifest["files"]["submission"] = copy_required(Path(args.submission), out_dir / "results.json")
    manifest["files"]["config"] = copy_required(Path(args.config), out_dir / "configs" / Path(args.config).name)
    manifest["files"]["metrics"] = copy_optional(Path(args.metrics) if args.metrics else None, out_dir / "reports" / "val_metrics.json")
    manifest["files"]["errors"] = copy_optional(Path(args.errors) if args.errors else None, out_dir / "reports" / "val_errors.csv")
    manifest["files"]["time_summary"] = copy_optional(Path(args.time_summary) if args.time_summary else None, out_dir / "reports" / "inference_time_summary.json")
    manifest["files"]["speed_buckets"] = copy_optional(Path(args.speed_buckets) if args.speed_buckets else None, out_dir / "reports" / "speed_buckets.json")
    manifest["files"]["technical_report"] = copy_optional(Path(args.report) if args.report else None, out_dir / Path(args.report).name if args.report else out_dir / "report.pdf")
    manifest["files"]["readme"] = copy_optional(Path("README.md"), out_dir / "README.md")
    manifest["files"]["architecture_doc"] = copy_optional(Path("docs/model_system_architecture.md"), out_dir / "docs" / "model_system_architecture.md")
    manifest["files"]["architecture_overview"] = copy_optional(Path("docs/model_architecture_overview.md"), out_dir / "docs" / "model_architecture_overview.md")
    if args.copy_docs:
        manifest["files"]["docs"] = copy_tree_files(
            Path("docs"),
            out_dir / "docs",
            patterns=("*.md", "*.svg", "*.png", "*.jpg", "*.jpeg"),
            recursive=True,
        )
    if args.copy_source:
        source_files: list[str] = []
        source_files.extend(copy_source_tree(Path("src"), out_dir / "source" / "src"))
        source_files.extend(copy_source_tree(Path("scripts"), out_dir / "source" / "scripts"))
        source_files.extend(copy_source_tree(Path("configs"), out_dir / "source" / "configs"))
        for filename in ["requirements.txt", "environment.yml", "README.md"]:
            copied = copy_optional(Path(filename), out_dir / "source" / filename)
            if copied:
                source_files.append(copied)
        manifest["files"]["source"] = source_files

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    write_reproduce_md(out_dir, manifest)
    print(f"Packaged delivery to {out_dir}")


if __name__ == "__main__":
    main()
