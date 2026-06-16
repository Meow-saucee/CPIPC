from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REF_CANDIDATE = [
    "yolo26n_ref_unionfloor05",
    "outputs/reports/submission_metrics_seg_ref_yolo26n_hybrid_unionfloor05_val.json",
    "outputs/reports/inference_time_summary_seg_ref_yolo26n_hybrid_unionfloor05.json",
    "outputs/reports/speed_buckets_seg_ref_yolo26n_hybrid_unionfloor05_test.json",
    "outputs/reports/delivery_audit_yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate.json",
]


def read_last_epoch(results_csv: Path) -> int:
    if not results_csv.exists():
        return -1
    with results_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return -1
    try:
        return int(float(rows[-1]["epoch"]))
    except (KeyError, ValueError):
        return -1


def run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] $ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def refresh_progress_json(root: Path, rel_run_dir: str, epochs: int, name: str, dry_run: bool) -> None:
    out_path = f"outputs/reports/training_progress_{name}.json"
    run(
        [
            sys.executable,
            "scripts/monitor_training.py",
            "--run-dir",
            rel_run_dir,
            "--epochs",
            str(epochs),
            "--json-out",
            out_path,
        ],
        cwd=root,
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for scale-combo training, then evaluate, package and compare.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", default="yolo11n_seg_scalecombo_best_candidate")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--project-root", default="/home/ruiyi/CPIPC/Dection")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    results_csv = run_dir / "results.csv"
    rel_run_dir = str(run_dir.relative_to(root)) if run_dir.is_relative_to(root) else str(run_dir)

    while True:
        epoch = read_last_epoch(results_csv)
        print(f"[{datetime.now().isoformat(timespec='seconds')}] training progress: epoch {epoch}/{args.epochs}", flush=True)
        if epoch >= 0:
            refresh_progress_json(root, rel_run_dir, args.epochs, args.name, args.dry_run)
        if epoch >= args.epochs:
            break
        if args.dry_run:
            print("dry-run: stop before waiting", flush=True)
            return
        time.sleep(max(1, args.poll_seconds))

    print(f"[{datetime.now().isoformat(timespec='seconds')}] training reached target epoch; start evaluation", flush=True)
    run(["bash", "scripts/eval_package_scalecombo.sh", rel_run_dir, args.name], cwd=root, dry_run=args.dry_run)

    scalecombo_candidate = [
        "yolo11n_scalecombo",
        f"outputs/reports/submission_metrics_{args.name}_val.json",
        f"outputs/reports/inference_time_summary_{args.name}.json",
        f"outputs/reports/speed_buckets_{args.name}_test.json",
        f"outputs/reports/delivery_audit_{args.name}.json",
    ]
    run(
        [
            sys.executable,
            "src/compare_candidates.py",
            "--candidate",
            *REF_CANDIDATE,
            "--candidate",
            *scalecombo_candidate,
            "--out-json",
            "outputs/reports/candidate_comparison.json",
            "--out-csv",
            "outputs/reports/candidate_comparison.csv",
            "--out-md",
            "outputs/reports/candidate_comparison.md",
        ],
        cwd=root,
        dry_run=args.dry_run,
    )
    print(f"[{datetime.now().isoformat(timespec='seconds')}] evaluation, packaging, comparison and archive finished", flush=True)

    run(
        [
            sys.executable,
            "src/archive_experiment.py",
            "--run-dir",
            rel_run_dir,
            "--config",
            "configs/yolo_seg_crack_hybrid.yaml",
            "--data-yaml",
            "data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml",
            "--model",
            "yolo11n-seg.pt",
            "--dataset-name",
            "cpipc-chip-crack-seg",
            "--imgsz",
            "1024",
            "--epochs",
            str(args.train_epochs),
            "--batch",
            "2",
            "--seed",
            "42",
            "--tag",
            "seg-scaleaware-scalecrop",
            "--metrics",
            f"outputs/reports/submission_metrics_{args.name}_val.json",
            "--errors",
            f"outputs/reports/submission_errors_{args.name}_val.csv",
            "--submission",
            f"outputs/submissions/results_{args.name}.json",
            "--command",
            "python src/train_yolo_seg.py --config configs/yolo_seg_crack_hybrid.yaml --data-yaml data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml --model /home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/yolo11n-seg.pt --imgsz 1024 --epochs 200 --batch 2 --device 0 --tag seg-scaleaware-scalecrop",
        ],
        cwd=root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
