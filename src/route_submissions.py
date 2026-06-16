from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from common import load_json, save_json


def load_sizes(dataset_root: Path) -> dict[int, tuple[int, int]]:
    sizes: dict[int, tuple[int, int]] = {}
    for split in ["trainval", "test"]:
        json_name = "trainval.json" if split == "trainval" else "test.json"
        json_path = dataset_root / split / json_name
        if not json_path.exists():
            continue
        for item in load_json(json_path).get("Dataset", []):
            image_id = int(item["ID"])
            if image_id in sizes:
                continue
            image_path = dataset_root / split / item["Image"]
            if image_path.exists():
                with Image.open(image_path) as im:
                    sizes[image_id] = (im.width, im.height)
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Route images between fast and accurate submission JSON files by image size.")
    parser.add_argument("--fast", required=True, help="Submission used for regular/small images.")
    parser.add_argument("--accurate", required=True, help="Submission used for large images.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--large-threshold", type=int, default=2048)
    parser.add_argument("--fast-min-side", type=int, default=0, help="Use fast submission only when max_side >= this value.")
    parser.add_argument("--fast-max-side", type=int, default=0, help="Use fast submission only when max_side <= this value; 0 disables upper bound.")
    args = parser.parse_args()

    fast_rows = load_json(args.fast)
    accurate_rows = load_json(args.accurate)
    if len(fast_rows) != len(accurate_rows):
        raise ValueError("fast and accurate submissions must have same row count")
    fast_by_id = {int(row["ID"]): row for row in fast_rows}
    accurate_by_id = {int(row["ID"]): row for row in accurate_rows}
    sizes = load_sizes(Path(args.dataset))

    out_rows: list[dict[str, Any]] = []
    fast_count = 0
    accurate_count = 0
    for row in fast_rows:
        image_id = int(row["ID"])
        if image_id not in accurate_by_id:
            raise ValueError(f"missing ID in accurate submission: {image_id}")
        width, height = sizes.get(image_id, (0, 0))
        max_side = max(width, height)
        in_fast_range = max_side >= args.fast_min_side and (args.fast_max_side <= 0 or max_side <= args.fast_max_side)
        if max_side > args.large_threshold:
            out_rows.append(dict(accurate_by_id[image_id]))
            accurate_count += 1
        elif in_fast_range:
            out_rows.append(dict(fast_by_id[image_id]))
            fast_count += 1
        else:
            out_rows.append(dict(accurate_by_id[image_id]))
            accurate_count += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_rows, out_path)
    print(f"Saved routed submission to {out_path}")
    print(f"rows={len(out_rows)}, fast={fast_count}, accurate={accurate_count}")


if __name__ == "__main__":
    main()
