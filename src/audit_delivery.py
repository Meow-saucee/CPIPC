from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import load_json, save_json


DEFAULT_REQUIRED_FILES = [
    "README.md",
    "REPRODUCE.md",
    "manifest.json",
    "results.json",
    "configs/yolo_seg_crack_hybrid.yaml",
    "reports/val_metrics.json",
    "reports/val_errors.csv",
    "reports/inference_time_summary.json",
    "reports/speed_buckets.json",
    "docs/model_system_architecture.md",
    "docs/model_framework_and_parameters.md",
    "docs/model_architecture_overview.md",
    "docs/parameter_map.md",
    "docs/technical_design_report.md",
    "docs/experiment_summary.md",
    "docs/final_delivery_checklist.md",
    "docs/speed_route_strategy.md",
    "docs/assets/system_pipeline.svg",
    "docs/assets/yolo_seg_architecture.svg",
    "docs/assets/inference_postprocess.svg",
    "source/src/infer_submit_seg.py",
    "source/src/eval_submission.py",
    "source/src/check_submit.py",
    "source/src/train_yolo_seg.py",
    "source/src/demote_oversized_boxes.py",
    "source/scripts/run_pipeline.py",
    "source/scripts/reproduce_final_speed_route.sh",
    "source/scripts/reproduce_ensemble_weighted.sh",
    "source/environment.yml",
    "source/requirements.txt",
]


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    severity: str = "error"


def add_check(checks: list[Check], name: str, passed: bool, detail: str, severity: str = "error") -> None:
    checks.append(Check(name=name, passed=passed, detail=detail, severity=severity))


def read_speed_buckets(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = load_json(path)
    if isinstance(data, dict):
        return data
    return {}


def get_bucket_avg_ms(bucket: dict[str, Any]) -> float | None:
    for key in ["avg_inference_time_ms", "avg_ms", "mean_ms", "avg"]:
        if key in bucket:
            return float(bucket[key])
    return None


def get_bucket_max_ms(bucket: dict[str, Any]) -> float | None:
    for key in ["max_inference_time_ms", "max_ms", "max"]:
        if key in bucket:
            return float(bucket[key])
    return None


def filtered_values(values: list[float], ignore_first: bool) -> list[float]:
    if ignore_first and len(values) > 1:
        return values[1:]
    return values


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "avg_ms": None, "max_ms": None, "min_ms": None}
    return {
        "count": len(values),
        "avg_ms": sum(values) / len(values),
        "max_ms": max(values),
        "min_ms": min(values),
    }


def bucket_stats_from_details(buckets: dict[str, Any], bucket_name: str, ignore_first: bool) -> dict[str, float | int | None] | None:
    details = buckets.get("details")
    if not isinstance(details, list):
        return None
    values = [
        float(item.get("inference_time_ms", 0.0))
        for item in details
        if isinstance(item, dict) and item.get("bucket") == bucket_name
    ]
    return summarize_values(filtered_values(values, ignore_first))


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return max(0, len(rows) - 1)


def run_submit_check(dataset: Path, submit: Path) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "src/check_submit.py",
        "--dataset",
        str(dataset),
        "--submit",
        str(submit),
    ]
    proc = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True, check=False)
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output


def check_files(delivery_dir: Path, checks: list[Check]) -> list[str]:
    missing: list[str] = []
    for rel in DEFAULT_REQUIRED_FILES:
        exists = (delivery_dir / rel).exists()
        add_check(checks, f"required_file:{rel}", exists, str(delivery_dir / rel) if exists else "missing")
        if not exists:
            missing.append(rel)
    pt_files = sorted((delivery_dir / "weights").glob("*.pt"))
    pth_files = sorted((delivery_dir / "weights").glob("*.pth"))
    add_check(checks, "weights_pt_exists", bool(pt_files), ", ".join(str(path) for path in pt_files) if pt_files else "missing *.pt")
    add_check(checks, "weights_pth_exists", bool(pth_files), ", ".join(str(path) for path in pth_files) if pth_files else "missing *.pth")
    if not pt_files:
        missing.append("weights/*.pt")
    if not pth_files:
        missing.append("weights/*.pth")
    return missing


def check_manifest(delivery_dir: Path, checks: list[Check]) -> dict[str, Any]:
    manifest_path = delivery_dir / "manifest.json"
    if not manifest_path.exists():
        add_check(checks, "manifest_readable", False, "manifest.json missing")
        return {}
    try:
        manifest = load_json(manifest_path)
    except Exception as exc:  # noqa: BLE001
        add_check(checks, "manifest_readable", False, f"cannot parse manifest.json: {exc}")
        return {}
    files = manifest.get("files", {})
    add_check(checks, "manifest_has_files_index", isinstance(files, dict) and bool(files), f"indexed keys={sorted(files) if isinstance(files, dict) else []}")
    extra_weights = files.get("extra_weights", []) if isinstance(files, dict) else []
    if extra_weights:
        missing_extra = [path for path in extra_weights if not Path(path).exists()]
        add_check(
            checks,
            "ensemble_extra_weights_exist",
            not missing_extra,
            f"extra_weights={len(extra_weights)}, missing={missing_extra}",
        )
        weighted_script = delivery_dir / "source" / "scripts" / "reproduce_ensemble_weighted.sh"
        calibrated_script = delivery_dir / "source" / "scripts" / "reproduce_ensemble_w075_calibrated.sh"
        final_speed_script = delivery_dir / "source" / "scripts" / "reproduce_final_speed_route.sh"
        if "route_regular_gt100_fastdetbox768_warm" in delivery_dir.name:
            expected_script = final_speed_script
        elif "w075_calibrated" in delivery_dir.name:
            expected_script = calibrated_script
        else:
            expected_script = weighted_script
        add_check(checks, "ensemble_reproduce_script_exists", expected_script.exists(), str(expected_script) if expected_script.exists() else "missing")
    return manifest


def check_metrics(
    metrics_path: Path,
    checks: list[Check],
    tiny_target: float,
    large_iou_target: float,
    large_iou_hard_target: bool,
) -> dict[str, Any]:
    if not metrics_path.exists():
        add_check(checks, "metrics_available", False, f"missing {metrics_path}")
        return {}
    metrics = load_json(metrics_path)
    tiny = float(metrics.get("tiny_recall_at_iou50", -1))
    large_matched = float(metrics.get("large_mean_matched_iou", -1))
    large_best = float(metrics.get("large_mean_best_iou", -1))
    map50 = float(metrics.get("mAP50", -1))
    recall = float(metrics.get("recall_at_iou50", -1))
    add_check(checks, "metrics_available", True, str(metrics_path))
    add_check(checks, "tiny_recall_target", tiny >= tiny_target, f"tiny_recall_at_iou50={tiny:.6f}, target>={tiny_target:.2f}")
    if large_iou_hard_target:
        add_check(
            checks,
            "large_matched_iou_target",
            large_matched >= large_iou_target,
            f"large_mean_matched_iou={large_matched:.6f}, target>={large_iou_target:.2f}, hard target",
            severity="error",
        )
        add_check(
            checks,
            "large_best_iou_reference",
            large_best >= large_iou_target,
            f"large_mean_best_iou={large_best:.6f}, reference target>={large_iou_target:.2f}",
            severity="warning",
        )
    else:
        add_check(
            checks,
            "large_matched_iou_reference_recorded",
            large_matched >= 0,
            f"large_mean_matched_iou={large_matched:.6f}, reference={large_iou_target:.2f}, not a delivery gate",
            severity="warning",
        )
        add_check(
            checks,
            "large_best_iou_reference_recorded",
            large_best >= 0,
            f"large_mean_best_iou={large_best:.6f}, reference={large_iou_target:.2f}, not a delivery gate",
            severity="warning",
        )
    add_check(checks, "map50_recorded", map50 >= 0, f"mAP50={map50:.6f}", severity="warning")
    add_check(checks, "recall_recorded", recall >= 0, f"recall_at_iou50={recall:.6f}", severity="warning")
    return metrics


def check_speed(
    summary_path: Path,
    buckets_path: Path,
    benchmark_path: Path,
    checks: list[Check],
    regular_target_ms: float,
    large_target_ms: float,
    ignore_first_timing: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary: dict[str, Any] = {}
    buckets: dict[str, Any] = {}
    if summary_path.exists():
        summary = load_json(summary_path)
        avg_ms = float(summary.get("avg_inference_time_ms", -1))
        max_ms = float(summary.get("max_inference_time_ms", -1))
        add_check(checks, "time_summary_available", True, str(summary_path))
        add_check(checks, "overall_avg_time_recorded", avg_ms >= 0, f"avg_inference_time_ms={avg_ms:.3f}", severity="warning")
        add_check(checks, "overall_max_time_recorded", max_ms >= 0, f"max_inference_time_ms={max_ms:.3f}", severity="warning")
    else:
        add_check(checks, "time_summary_available", False, f"missing {summary_path}")

    benchmark = load_json(benchmark_path) if benchmark_path.exists() else {}
    benchmark_images = benchmark.get("images") if isinstance(benchmark, dict) else None
    benchmark_regular_max = None
    if isinstance(benchmark_images, list) and benchmark_images:
        regular_values = [
            float(item["max_ms"])
            for item in benchmark_images
            if isinstance(item, dict) and float(item.get("max_side", 0)) <= 2048 and item.get("max_ms") is not None
        ]
        if regular_values:
            benchmark_regular_max = max(regular_values)
            add_check(
                checks,
                "regular_image_benchmark_speed_target",
                benchmark_regular_max < regular_target_ms,
                f"benchmark regular max={benchmark_regular_max}ms, target<{regular_target_ms:.0f}ms, source={benchmark_path}",
            )
    elif benchmark_path.exists():
        add_check(checks, "regular_image_benchmark_speed_target", False, f"benchmark file unreadable or empty: {benchmark_path}")

    if buckets_path.exists():
        buckets = read_speed_buckets(buckets_path)
        add_check(checks, "speed_buckets_available", True, str(buckets_path))
        regular_bucket = buckets.get("regular") or buckets.get("regular_images") or buckets.get("small")
        large_bucket = buckets.get("large") or buckets.get("large_images") or buckets.get("huge")
        if isinstance(regular_bucket, dict):
            regular_avg = get_bucket_avg_ms(regular_bucket)
            regular_max = get_bucket_max_ms(regular_bucket)
            regular_filtered = bucket_stats_from_details(buckets, "regular", ignore_first_timing)
            if regular_filtered:
                regular_avg = regular_filtered["avg_ms"] if regular_filtered["avg_ms"] is not None else regular_avg
                regular_max = regular_filtered["max_ms"] if regular_filtered["max_ms"] is not None else regular_max
            regular_avg_passed = regular_avg is not None and float(regular_avg) < regular_target_ms
            regular_max_passed = regular_max is not None and float(regular_max) < regular_target_ms
            regular_max_severity = "warning" if benchmark_regular_max is not None and benchmark_regular_max < regular_target_ms else "error"
            add_check(checks, "regular_image_avg_speed_target", regular_avg_passed, f"regular avg={regular_avg}ms, target<{regular_target_ms:.0f}ms")
            add_check(
                checks,
                "regular_image_single_speed_target",
                regular_max_passed,
                f"regular max={regular_max}ms, target<{regular_target_ms:.0f}ms, ignore_first_timing={ignore_first_timing}",
                severity=regular_max_severity,
            )
        else:
            add_check(checks, "regular_image_avg_speed_target", False, "regular speed bucket missing")
            add_check(checks, "regular_image_single_speed_target", False, "regular speed bucket missing")
        if isinstance(large_bucket, dict):
            large_avg = get_bucket_avg_ms(large_bucket)
            large_max = get_bucket_max_ms(large_bucket)
            large_filtered = bucket_stats_from_details(buckets, "large", False)
            if large_filtered:
                large_avg = large_filtered["avg_ms"] if large_filtered["avg_ms"] is not None else large_avg
                large_max = large_filtered["max_ms"] if large_filtered["max_ms"] is not None else large_max
            large_avg_passed = large_avg is not None and float(large_avg) < large_target_ms
            large_max_passed = large_max is not None and float(large_max) < large_target_ms
            add_check(checks, "large_image_avg_speed_target", large_avg_passed, f"large avg={large_avg}ms, target<{large_target_ms:.0f}ms")
            add_check(checks, "large_image_single_speed_target", large_max_passed, f"large max={large_max}ms, target<{large_target_ms:.0f}ms")
        else:
            add_check(checks, "large_image_avg_speed_target", False, "large speed bucket missing")
            add_check(checks, "large_image_single_speed_target", False, "large speed bucket missing")
    else:
        add_check(checks, "speed_buckets_available", False, f"missing {buckets_path}")
    return summary, buckets


def write_markdown(report: dict[str, Any], path: Path) -> None:
    checks = report["checks"]
    passed = sum(1 for item in checks if item["passed"])
    total = len(checks)
    hard_failures = [item for item in checks if not item["passed"] and item["severity"] == "error"]
    warnings = [item for item in checks if not item["passed"] and item["severity"] == "warning"]
    lines = [
        "# 交付验收审计报告",
        "",
        f"- 交付目录：`{report['delivery_dir']}`",
        f"- 总体状态：`{report['status']}`",
        f"- 检查项：{passed}/{total} 通过",
        f"- 硬失败：{len(hard_failures)}",
        f"- 警告：{len(warnings)}",
        "",
        "## 关键指标",
        "",
    ]
    metrics = report.get("metrics") or {}
    if metrics:
        lines.extend(
            [
                f"- mAP50：{metrics.get('mAP50')}",
                f"- Recall@IoU50：{metrics.get('recall_at_iou50')}",
                f"- Tiny Recall@IoU50：{metrics.get('tiny_recall_at_iou50')}",
                f"- Large Matched IoU：{metrics.get('large_mean_matched_iou')}",
                f"- Large Best IoU：{metrics.get('large_mean_best_iou')}",
                "",
            ]
        )
    speed = report.get("speed_summary") or {}
    if speed:
        lines.extend(
            [
                "## 推理耗时",
                "",
                f"- 测试图像数：{speed.get('images')}",
                f"- 平均耗时：{speed.get('avg_inference_time_ms')} ms",
                f"- 最大耗时：{speed.get('max_inference_time_ms')} ms",
                "",
            ]
        )
    lines.extend(["## 未通过项", ""])
    if hard_failures or warnings:
        for item in hard_failures + warnings:
            lines.append(f"- `{item['name']}` [{item['severity']}]: {item['detail']}")
    else:
        lines.append("- 无")
    lines.extend(["", "## 全部检查项", ""])
    for item in checks:
        mark = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- `{mark}` `{item['name']}`：{item['detail']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit delivery package completeness, submission validity and metric targets.")
    parser.add_argument("--delivery", default="deliverables/yolo26n_seg_ref_hybrid_unionfloor05_iou_candidate")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--tiny-target", type=float, default=0.90)
    parser.add_argument("--large-iou-target", type=float, default=0.85, help="Reference value only unless --large-iou-hard-target is set.")
    parser.add_argument(
        "--large-iou-hard-target",
        action="store_true",
        help="Treat large matched IoU as a hard failure instead of a reference warning.",
    )
    parser.add_argument("--regular-time-ms", type=float, default=100.0)
    parser.add_argument("--large-time-ms", type=float, default=2000.0)
    parser.add_argument("--ignore-first-timing", action="store_true", help="Ignore the first regular-image timing when checking strict per-image speed.")
    parser.add_argument("--benchmark-speed", default=None, help="Optional real benchmark JSON from src/benchmark_inference.py.")
    parser.add_argument("--out-json", default="outputs/reports/delivery_audit.json")
    parser.add_argument("--out-md", default="outputs/reports/delivery_audit.md")
    args = parser.parse_args()

    delivery_dir = Path(args.delivery)
    checks: list[Check] = []
    add_check(checks, "delivery_dir_exists", delivery_dir.exists() and delivery_dir.is_dir(), str(delivery_dir))
    if not delivery_dir.exists():
        report = {
            "delivery_dir": str(delivery_dir),
            "status": "FAIL",
            "checks": [check.__dict__ for check in checks],
        }
        save_json(report, args.out_json)
        write_markdown(report, Path(args.out_md))
        raise SystemExit(1)

    missing = check_files(delivery_dir, checks)
    manifest = check_manifest(delivery_dir, checks)
    metrics = check_metrics(
        delivery_dir / "reports" / "val_metrics.json",
        checks,
        args.tiny_target,
        args.large_iou_target,
        args.large_iou_hard_target,
    )
    speed_summary, speed_buckets = check_speed(
        delivery_dir / "reports" / "inference_time_summary.json",
        delivery_dir / "reports" / "speed_buckets.json",
        Path(args.benchmark_speed) if args.benchmark_speed else delivery_dir / "reports" / "benchmark_speed.json",
        checks,
        args.regular_time_ms,
        args.large_time_ms,
        args.ignore_first_timing,
    )
    errors_rows = count_csv_rows(delivery_dir / "reports" / "val_errors.csv")
    add_check(checks, "val_error_csv_readable", errors_rows is not None, f"rows={errors_rows}")

    submit_ok, submit_output = run_submit_check(Path(args.dataset), delivery_dir / "results.json")
    add_check(checks, "submission_schema_valid", submit_ok, submit_output)

    hard_failures = [check for check in checks if not check.passed and check.severity == "error"]
    status = "PASS" if not hard_failures else "FAIL"
    report = {
        "delivery_dir": str(delivery_dir),
        "status": status,
        "missing_files": missing,
        "manifest": manifest,
        "metrics": metrics,
        "speed_summary": speed_summary,
        "speed_buckets": speed_buckets,
        "checks": [check.__dict__ for check in checks],
    }
    save_json(report, args.out_json)
    write_markdown(report, Path(args.out_md))
    print(f"Delivery audit status: {status}")
    print(f"JSON report: {args.out_json}")
    print(f"Markdown report: {args.out_md}")
    if hard_failures:
        print("Hard failures:")
        for check in hard_failures:
            print(f" - {check.name}: {check.detail}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
