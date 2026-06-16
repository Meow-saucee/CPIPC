from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG = "configs/yolo_seg_crack_hybrid.yaml"
DEFAULT_REF_WEIGHTS = "/home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/runs/yolo26n_seg_baseline-2/weights/best.pt"
DEFAULT_PRETRAINED_SEG = "/home/ruiyi/CPIPC/跨尺度芯片图像的裂纹缺陷智能检测算法设计/yolo11n-seg.pt"
DEFAULT_SUBMISSION = "outputs/submissions/results_seg_ref_yolo26n_hybrid_unionfloor05.json"
DEFAULT_DELIVERY = "yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate"


def run(cmd: list[str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def stage_prepare(args: argparse.Namespace) -> list[list[str]]:
    return [
        [sys.executable, "src/prepare_yolo_seg.py", "--config", args.config],
        [
            sys.executable,
            "src/build_scale_aware_train_list.py",
            "--config",
            args.config,
            "--out-suffix",
            "scaleaware",
            "--tiny-repeat",
            "3",
            "--large-repeat",
            "3",
            "--huge-repeat",
            "2",
            "--tiny-large-repeat",
            "4",
            "--max-repeat",
            "4",
        ],
        [
            sys.executable,
            "src/build_scale_crop_dataset.py",
            "--config",
            args.config,
            "--out-suffix",
            "scalecrop",
            "--crop-size",
            "1024",
            "--context",
            "2.5",
            "--tiny-repeat",
            "2",
            "--large-repeat",
            "1",
            "--max-crops-per-image",
            "4",
        ],
        [
            sys.executable,
            "src/build_combined_yolo_data.py",
            "--config",
            args.config,
            "--out-suffix",
            "scaleaware_scalecrop",
            "--train-lists",
            "data/yolo_seg/train_scaleaware.txt",
            "data/yolo_seg/train_scalecrop_only.txt",
        ],
    ]


def stage_check(args: argparse.Namespace) -> list[list[str]]:
    return [
        [
            sys.executable,
            "src/check_yolo_seg_data.py",
            "--data-yaml",
            args.train_data_yaml,
            "--out",
            "outputs/reports/check_crack_seg_scaleaware_scalecrop.json",
        ]
    ]


def stage_smoke(args: argparse.Namespace) -> list[list[str]]:
    return [
        [
            sys.executable,
            "src/train_yolo_seg.py",
            "--config",
            args.config,
            "--data-yaml",
            args.train_data_yaml,
            "--model",
            args.pretrained_seg,
            "--imgsz",
            "640",
            "--epochs",
            "1",
            "--batch",
            "1",
            "--device",
            args.device,
            "--name",
            "yolo11n_seg_scalecombo_smoke",
            "--tag",
            "seg-scalecombo-smoke",
            "--no-archive",
        ]
    ]


def stage_train(args: argparse.Namespace) -> list[list[str]]:
    return [
        [
            sys.executable,
            "src/train_yolo_seg.py",
            "--config",
            args.config,
            "--data-yaml",
            args.train_data_yaml,
            "--model",
            args.pretrained_seg,
            "--imgsz",
            str(args.train_imgsz),
            "--epochs",
            str(args.epochs),
            "--batch",
            str(args.batch),
            "--device",
            args.device,
            "--tag",
            "seg-scaleaware-scalecrop",
        ]
    ]


def stage_eval_ref(args: argparse.Namespace) -> list[list[str]]:
    return [
        [
            sys.executable,
            "src/infer_submit_seg.py",
            "--config",
            args.config,
            "--weights",
            args.weights,
            "--split",
            "val",
            "--out",
            "outputs/submissions/val_pred_pipeline_ref.json",
        ],
        [
            sys.executable,
            "src/eval_submission.py",
            "--config",
            args.config,
            "--submit",
            "outputs/submissions/val_pred_pipeline_ref.json",
            "--split",
            "val",
            "--out",
            "outputs/reports/pipeline_ref_val_metrics.json",
            "--errors",
            "outputs/reports/pipeline_ref_val_errors.csv",
        ],
    ]


def stage_submit_ref(args: argparse.Namespace) -> list[list[str]]:
    return [
        [
            sys.executable,
            "src/infer_submit_seg.py",
            "--config",
            args.config,
            "--weights",
            args.weights,
            "--split",
            "test",
            "--out",
            args.submission,
        ],
        [sys.executable, "src/check_submit.py", "--dataset", "dataset", "--submit", args.submission],
    ]


def stage_package(args: argparse.Namespace) -> list[list[str]]:
    return [
        [
            sys.executable,
            "src/package_delivery.py",
            "--name",
            args.delivery_name,
            "--weights",
            args.weights,
            "--submission",
            args.submission,
            "--config",
            args.config,
            "--metrics",
            "outputs/reports/submission_metrics_seg_ref_yolo26n_hybrid_unionfloor05_val.json",
            "--errors",
            "outputs/reports/submission_errors_seg_ref_yolo26n_hybrid_unionfloor05_val.csv",
            "--time-summary",
            "outputs/reports/inference_time_summary_seg_ref_yolo26n_hybrid_unionfloor05.json",
            "--speed-buckets",
            "outputs/reports/speed_buckets_seg_ref_yolo26n_hybrid_unionfloor05_test.json",
            "--report",
            "docs/technical_design_report.md",
            "--copy-docs",
            "--copy-source",
        ]
    ]


def stage_audit(args: argparse.Namespace) -> list[list[str]]:
    delivery_dir = str(Path("deliverables") / args.delivery_name)
    return [
        [
            sys.executable,
            "src/audit_delivery.py",
            "--delivery",
            delivery_dir,
            "--dataset",
            "dataset",
            "--out-json",
            f"outputs/reports/delivery_audit_{args.delivery_name}.json",
            "--out-md",
            f"outputs/reports/delivery_audit_{args.delivery_name}.md",
        ]
    ]


STAGES = {
    "prepare": stage_prepare,
    "check": stage_check,
    "smoke": stage_smoke,
    "train": stage_train,
    "eval-ref": stage_eval_ref,
    "submit-ref": stage_submit_ref,
    "package": stage_package,
    "audit": stage_audit,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible CPIPC crack pipeline stages.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_REF_WEIGHTS, help="Weights for reference inference/package stages.")
    parser.add_argument("--pretrained-seg", default=DEFAULT_PRETRAINED_SEG, help="Pretrained YOLO-seg weights for training stages.")
    parser.add_argument("--train-data-yaml", default="data/yolo_seg/crack_seg_scaleaware_scalecrop.yaml")
    parser.add_argument("--submission", default=DEFAULT_SUBMISSION)
    parser.add_argument("--delivery-name", default=DEFAULT_DELIVERY)
    parser.add_argument("--device", default="0")
    parser.add_argument("--train-imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--stages", nargs="+", default=["check"], choices=sorted(STAGES))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    if not Path(args.config).exists():
        raise FileNotFoundError(args.config)

    for stage in args.stages:
        print(f"\n=== Stage: {stage} ===", flush=True)
        for cmd in STAGES[stage](args):
            run(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
