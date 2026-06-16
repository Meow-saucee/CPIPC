from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-image inference time from a submission JSON.")
    parser.add_argument("--submit", default="outputs/submissions/results.json")
    parser.add_argument("--out-json", default="outputs/reports/inference_time_summary.json")
    parser.add_argument("--out-csv", default="outputs/reports/inference_time_per_image.csv")
    args = parser.parse_args()

    rows = load_json(args.submit)
    times = [float(row.get("inference_time_ms", 0.0)) for row in rows]
    summary = {
        "submission": args.submit,
        "images": len(rows),
        "avg_inference_time_ms": sum(times) / len(times) if times else 0.0,
        "max_inference_time_ms": max(times) if times else 0.0,
        "min_inference_time_ms": min(times) if times else 0.0,
    }
    save_json(summary, args.out_json)

    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "image path", "inference_time_ms", "pred_count"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ID": row.get("ID"),
                    "image path": row.get("image path"),
                    "inference_time_ms": row.get("inference_time_ms"),
                    "pred_count": len(row.get("predict_bboxes", [])),
                }
            )
    print(f"Saved summary to {args.out_json}")
    print(f"Saved per-image times to {args.out_csv}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
