from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from common import clip_xyxy, load_json, save_json


def box_dims(pred: dict[str, Any]) -> tuple[float, float, float]:
    width = max(0.0, float(pred["x2"]) - float(pred["x1"]))
    height = max(0.0, float(pred["y2"]) - float(pred["y1"]))
    area = width * height
    return width, height, area


def aspect(width: float, height: float) -> float:
    return max(width, height) / max(1e-6, min(width, height))


def calibrate_box(
    pred: dict[str, Any],
    image_size: tuple[int, int],
    scale_x: float,
    scale_y: float,
    long_scale: float,
    short_scale: float,
    shift_x: float,
    shift_y: float,
) -> None:
    width, height = image_size
    x1, y1, x2, y2 = map(float, [pred["x1"], pred["y1"], pred["x2"], pred["y2"]])
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0 + shift_x * box_w
    cy = (y1 + y2) / 2.0 + shift_y * box_h
    sx = scale_x
    sy = scale_y
    if box_w >= box_h:
        sx *= long_scale
        sy *= short_scale
    else:
        sx *= short_scale
        sy *= long_scale
    new_w = max(1.0, box_w * sx)
    new_h = max(1.0, box_h * sy)
    clipped = clip_xyxy([cx - new_w / 2.0, cy - new_h / 2.0, cx + new_w / 2.0, cy + new_h / 2.0], width, height)
    pred["x1"], pred["y1"], pred["x2"], pred["y2"] = clipped


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
            if not image_path.exists():
                continue
            with Image.open(image_path) as im:
                sizes[image_id] = (im.width, im.height)
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Directly calibrate large or elongated boxes in a submission JSON.")
    parser.add_argument("--submit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--min-area", type=float, default=90000.0)
    parser.add_argument("--min-side", type=float, default=700.0)
    parser.add_argument("--min-aspect", type=float, default=3.0)
    parser.add_argument("--score-min", type=float, default=0.0)
    parser.add_argument("--scale-x", type=float, default=1.0)
    parser.add_argument("--scale-y", type=float, default=1.0)
    parser.add_argument("--long-scale", type=float, default=1.0)
    parser.add_argument("--short-scale", type=float, default=1.0)
    parser.add_argument("--shift-x", type=float, default=0.0)
    parser.add_argument("--shift-y", type=float, default=0.0)
    args = parser.parse_args()

    rows = load_json(args.submit)
    sizes = load_sizes(Path(args.dataset))
    changed = 0
    for row in rows:
        image_id = int(row["ID"])
        size = sizes.get(image_id)
        if size is None:
            continue
        for pred in row.get("predict_bboxes", []):
            score = float(pred.get("score", 0.0))
            if score < args.score_min:
                continue
            box_w, box_h, area = box_dims(pred)
            if area < args.min_area and not (max(box_w, box_h) >= args.min_side and aspect(box_w, box_h) >= args.min_aspect):
                continue
            calibrate_box(
                pred,
                size,
                scale_x=args.scale_x,
                scale_y=args.scale_y,
                long_scale=args.long_scale,
                short_scale=args.short_scale,
                shift_x=args.shift_x,
                shift_y=args.shift_y,
            )
            changed += 1
        row["predict_bboxes"] = sorted(row.get("predict_bboxes", []), key=lambda item: float(item.get("score", 0.0)), reverse=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(rows, out_path)
    print(f"Saved calibrated submission to {out_path}")
    print(f"rows={len(rows)}, changed_boxes={changed}")


if __name__ == "__main__":
    main()
