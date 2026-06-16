from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from common import load_json, save_json


def image_root(dataset: Path, split: str) -> Path:
    if split == "test":
        return dataset / "test"
    if split in {"trainval", "val"}:
        return dataset / "trainval"
    raise ValueError(f"Unsupported split: {split}")


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "avg_ms": None, "max_ms": None, "min_ms": None}
    return {
        "count": len(values),
        "avg_ms": sum(values) / len(values),
        "max_ms": max(values),
        "min_ms": min(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze submission inference speed by image scale buckets.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--submit", required=True)
    parser.add_argument("--split", choices=["test", "trainval", "val"], default="test")
    parser.add_argument("--out", default=None)
    parser.add_argument("--regular-max-side", type=int, default=2048)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    root = image_root(dataset, args.split)
    rows = load_json(args.submit)
    buckets: dict[str, list[float]] = {"regular": [], "large": []}
    details: list[dict[str, Any]] = []

    for row in rows:
        rel_path = row["image path"]
        path = root / rel_path
        with Image.open(path) as im:
            width, height = im.width, im.height
        max_side = max(width, height)
        bucket = "regular" if max_side <= args.regular_max_side else "large"
        elapsed = float(row.get("inference_time_ms", 0.0))
        buckets[bucket].append(elapsed)
        details.append(
            {
                "ID": row["ID"],
                "image": rel_path,
                "width": width,
                "height": height,
                "max_side": max_side,
                "bucket": bucket,
                "inference_time_ms": elapsed,
                "num_preds": len(row.get("predict_bboxes", [])),
            }
        )

    summary = {
        "submission": args.submit,
        "split": args.split,
        "regular_max_side": args.regular_max_side,
        "overall": summarize([float(row.get("inference_time_ms", 0.0)) for row in rows]),
        "regular": summarize(buckets["regular"]),
        "large": summarize(buckets["large"]),
        "details": details,
    }
    out_path = Path(args.out) if args.out else Path("outputs/reports/submission_speed_buckets.json")
    save_json(summary, out_path)
    print(f"Saved speed bucket report to {out_path}")
    print({key: summary[key] for key in ["overall", "regular", "large"]})


if __name__ == "__main__":
    main()
