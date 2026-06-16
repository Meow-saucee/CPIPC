from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(results_csv: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key.strip()] = float(value)
                except (TypeError, ValueError):
                    continue
            if "epoch" in parsed:
                rows.append(parsed)
    return rows


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def metric(row: dict[str, float], key: str) -> float:
    return float(row.get(key, 0.0))


def box_summary(row: dict[str, float], prefix: str) -> dict[str, float | int]:
    return {
        "epoch": int(metric(row, "epoch")),
        "precision": metric(row, f"metrics/precision({prefix})"),
        "recall": metric(row, f"metrics/recall({prefix})"),
        "mAP50": metric(row, f"metrics/mAP50({prefix})"),
        "mAP50_95": metric(row, f"metrics/mAP50-95({prefix})"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an Ultralytics results.csv while training is running.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--json-out", default=None, help="Optional path for machine-readable progress summary.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise SystemExit(f"Missing results.csv: {results_csv}")

    rows = read_rows(results_csv)
    if not rows:
        raise SystemExit(f"No rows yet: {results_csv}")

    last = rows[-1]
    best_box = max(rows, key=lambda item: metric(item, "metrics/mAP50(B)"))
    best_mask = max(rows, key=lambda item: metric(item, "metrics/mAP50(M)"))
    last_epoch = int(metric(last, "epoch"))
    elapsed = metric(last, "time")
    sec_per_epoch = elapsed / max(1, last_epoch)
    remaining_epochs = max(0, args.epochs - last_epoch)
    summary = {
        "run_dir": str(run_dir),
        "current_epoch": last_epoch,
        "target_epochs": args.epochs,
        "elapsed_seconds": elapsed,
        "avg_epoch_seconds": sec_per_epoch,
        "eta_seconds": sec_per_epoch * remaining_epochs,
        "last_box": box_summary(last, "B"),
        "best_box": box_summary(best_box, "B"),
        "best_mask": box_summary(best_mask, "M"),
    }

    print(f"run_dir: {run_dir}")
    print(f"progress: epoch {last_epoch}/{args.epochs}")
    print(f"elapsed: {fmt_time(elapsed)}")
    print(f"avg_epoch_time: {fmt_time(sec_per_epoch)}")
    print(f"eta: {fmt_time(sec_per_epoch * remaining_epochs)}")
    print(
        "last_box: "
        f"P={metric(last, 'metrics/precision(B)'):.4f}, "
        f"R={metric(last, 'metrics/recall(B)'):.4f}, "
        f"mAP50={metric(last, 'metrics/mAP50(B)'):.4f}, "
        f"mAP50-95={metric(last, 'metrics/mAP50-95(B)'):.4f}"
    )
    print(
        "best_box: "
        f"epoch={int(metric(best_box, 'epoch'))}, "
        f"P={metric(best_box, 'metrics/precision(B)'):.4f}, "
        f"R={metric(best_box, 'metrics/recall(B)'):.4f}, "
        f"mAP50={metric(best_box, 'metrics/mAP50(B)'):.4f}, "
        f"mAP50-95={metric(best_box, 'metrics/mAP50-95(B)'):.4f}"
    )
    print(
        "best_mask: "
        f"epoch={int(metric(best_mask, 'epoch'))}, "
        f"P={metric(best_mask, 'metrics/precision(M)'):.4f}, "
        f"R={metric(best_mask, 'metrics/recall(M)'):.4f}, "
        f"mAP50={metric(best_mask, 'metrics/mAP50(M)'):.4f}, "
        f"mAP50-95={metric(best_mask, 'metrics/mAP50-95(M)'):.4f}"
    )
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"saved_json: {out_path}")


if __name__ == "__main__":
    main()
