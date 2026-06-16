from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from common import load_json, save_json


METRIC_KEYS = [
    "mAP50",
    "precision_at_conf",
    "recall_at_iou50",
    "tiny_recall_at_iou50",
    "large_mean_matched_iou",
    "large_mean_best_iou",
    "predicted_boxes",
]

TINY_RECALL_TARGET = 0.90
LARGE_IOU_TARGET = 0.85  # reference only
REGULAR_TIME_TARGET_MS = 100.0
LARGE_TIME_TARGET_MS = 2000.0


def read_optional_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {"_missing": str(p)}
    data = load_json(p)
    return data if isinstance(data, dict) else {"_invalid": str(p)}


def get_nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def flatten_candidate(
    name: str,
    metrics: dict[str, Any],
    time_summary: dict[str, Any],
    speed_buckets: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name}
    for key in METRIC_KEYS:
        row[key] = metrics.get(key)

    row["overall_avg_ms"] = time_summary.get("avg_inference_time_ms") or get_nested(speed_buckets, "overall", "avg_ms")
    row["overall_max_ms"] = time_summary.get("max_inference_time_ms") or get_nested(speed_buckets, "overall", "max_ms")
    row["regular_avg_ms"] = get_nested(speed_buckets, "regular", "avg_ms")
    row["regular_max_ms"] = get_nested(speed_buckets, "regular", "max_ms")
    row["large_avg_ms"] = get_nested(speed_buckets, "large", "avg_ms")
    row["large_max_ms"] = get_nested(speed_buckets, "large", "max_ms")
    row["audit_status"] = audit.get("status")

    tiny = row.get("tiny_recall_at_iou50")
    large_iou = row.get("large_mean_matched_iou")
    regular_max = row.get("regular_max_ms")
    large_max = row.get("large_max_ms")
    row["tiny_recall_pass"] = isinstance(tiny, (int, float)) and float(tiny) >= TINY_RECALL_TARGET
    row["large_iou_pass"] = isinstance(large_iou, (int, float)) and float(large_iou) >= LARGE_IOU_TARGET
    row["regular_speed_pass"] = isinstance(regular_max, (int, float)) and float(regular_max) < REGULAR_TIME_TARGET_MS
    row["large_speed_pass"] = isinstance(large_max, (int, float)) and float(large_max) < LARGE_TIME_TARGET_MS
    row["hard_targets_pass"] = all(
        [
            row["tiny_recall_pass"],
            row["regular_speed_pass"],
            row["large_speed_pass"],
        ]
    )

    failed = []
    for item in audit.get("checks", []):
        if isinstance(item, dict) and item.get("severity", "error") == "error" and not item.get("passed", False):
            failed.append(item.get("name", "unknown"))
    row["audit_failed"] = ";".join(failed)
    return row


def candidate_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    hard = 1.0 if row.get("hard_targets_pass") else 0.0
    tiny = float(row.get("tiny_recall_at_iou50") or -1.0)
    large_iou = float(row.get("large_mean_matched_iou") or -1.0)
    map50 = float(row.get("mAP50") or -1.0)
    regular_max = row.get("regular_max_ms")
    speed_penalty = -float(regular_max) if isinstance(regular_max, (int, float)) else float("-inf")
    return hard, tiny, map50, large_iou, speed_penalty


def recommend_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=candidate_rank_key)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        *METRIC_KEYS,
        "overall_avg_ms",
        "overall_max_ms",
        "regular_avg_ms",
        "regular_max_ms",
        "large_avg_ms",
        "large_max_ms",
        "tiny_recall_pass",
        "large_iou_pass",
        "regular_speed_pass",
        "large_speed_pass",
        "hard_targets_pass",
        "audit_status",
        "audit_failed",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recommended = recommend_candidate(rows)
    columns = [
        "name",
        "mAP50",
        "recall_at_iou50",
        "tiny_recall_at_iou50",
        "large_mean_matched_iou",
        "large_mean_best_iou",
        "regular_max_ms",
        "large_max_ms",
        "hard_targets_pass",
        "audit_status",
    ]
    lines = ["# Candidate Comparison", ""]
    if recommended:
        lines.extend(
            [
                "## Recommendation",
                "",
                f"- Recommended candidate: `{recommended.get('name')}`",
                f"- Hard targets pass: `{recommended.get('hard_targets_pass')}`",
                f"- Tiny recall pass: `{recommended.get('tiny_recall_pass')}`",
                f"- Large IoU reference pass: `{recommended.get('large_iou_pass')}`",
                f"- Regular speed pass: `{recommended.get('regular_speed_pass')}`",
                f"- Large speed pass: `{recommended.get('large_speed_pass')}`",
                "",
            ]
        )
    lines.extend(["## Metrics", "", "|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"])
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("|" + "|".join(values) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate metrics, speed and audit reports.")
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=5,
        metavar=("NAME", "METRICS_JSON", "TIME_JSON", "SPEED_JSON", "AUDIT_JSON"),
        required=True,
        help="Candidate tuple. Use 'none' for missing optional files.",
    )
    parser.add_argument("--out-json", default="outputs/reports/candidate_comparison.json")
    parser.add_argument("--out-csv", default="outputs/reports/candidate_comparison.csv")
    parser.add_argument("--out-md", default="outputs/reports/candidate_comparison.md")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for name, metrics_path, time_path, speed_path, audit_path in args.candidate:
        metrics = read_optional_json(None if metrics_path == "none" else metrics_path)
        time_summary = read_optional_json(None if time_path == "none" else time_path)
        speed_buckets = read_optional_json(None if speed_path == "none" else speed_path)
        audit = read_optional_json(None if audit_path == "none" else audit_path)
        rows.append(flatten_candidate(name, metrics, time_summary, speed_buckets, audit))

    save_json({"candidates": rows}, args.out_json)
    write_csv(rows, Path(args.out_csv))
    write_markdown(rows, Path(args.out_md))
    print(f"Saved comparison to {args.out_json}, {args.out_csv}, {args.out_md}")
    for row in rows:
        print(
            row["name"],
            {
                "mAP50": row.get("mAP50"),
                "tiny_recall": row.get("tiny_recall_at_iou50"),
                "large_iou": row.get("large_mean_matched_iou"),
                "regular_max_ms": row.get("regular_max_ms"),
                "audit": row.get("audit_status"),
            },
        )


if __name__ == "__main__":
    main()
