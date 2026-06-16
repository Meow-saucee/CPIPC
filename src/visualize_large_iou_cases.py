from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from common import save_json


def parse_box(value: str) -> list[float]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or len(parsed) != 4:
        raise ValueError(f"Invalid box: {value}")
    return [float(v) for v in parsed]


def scale_box(box: list[float], scale: float) -> list[float]:
    return [v * scale for v in box]


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str) -> None:
    x, y = xy
    font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    rect = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
    draw.rectangle(rect, fill=fill)
    draw.text((x, y), text, fill="white", font=font)


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: str, label: str, width: int = 3) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    draw_label(draw, (x1, max(0, y1 - 14)), label, color)


def load_rows(path: Path, limit: int | None, max_iou: float | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["matched_iou_float"] = float(row["matched_iou"])
            row["best_iou_float"] = float(row["best_iou"])
            if max_iou is not None and row["matched_iou_float"] > max_iou:
                continue
            rows.append(row)
    rows.sort(key=lambda item: item["matched_iou_float"])
    return rows[:limit] if limit else rows


def make_canvas(image_path: Path, max_side: int) -> tuple[Image.Image, float]:
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    scale = min(1.0, max_side / max(image.width, image.height))
    if scale < 1.0:
        new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
    return image, scale


def visualize_row(row: dict[str, Any], dataset_root: Path, out_path: Path, max_side: int) -> dict[str, Any]:
    image_path = dataset_root / "trainval" / row["image"]
    image, scale = make_canvas(image_path, max_side=max_side)
    draw = ImageDraw.Draw(image)
    gt_box = parse_box(row["gt_box"])
    matched_box = parse_box(row["matched_box"])
    best_box = parse_box(row["best_box"])

    draw_box(draw, scale_box(gt_box, scale), "lime", "GT", width=3)
    if matched_box[2] > matched_box[0] and matched_box[3] > matched_box[1]:
        draw_box(draw, scale_box(matched_box, scale), "red", f"matched {float(row['matched_iou']):.3f}", width=3)
    if best_box[2] > best_box[0] and best_box[3] > best_box[1]:
        draw_box(draw, scale_box(best_box, scale), "dodgerblue", f"best {float(row['best_iou']):.3f}", width=3)

    info = [
        f"ID={row['image_id']} ann={row['ann_idx']} {row['image']}",
        f"failure={row['failure_type']}",
        f"matched_iou={float(row['matched_iou']):.4f} score={float(row['matched_score']):.4f}",
        f"best_iou={float(row['best_iou']):.4f} score={float(row['best_score']):.4f}",
        f"gt_w={float(row['gt_w']):.1f} gt_h={float(row['gt_h']):.1f}",
    ]
    y = 8
    for text in info:
        draw_label(draw, (8, y), text, "black")
        y += 17

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return {
        "image_id": int(row["image_id"]),
        "image": row["image"],
        "ann_idx": int(row["ann_idx"]),
        "failure_type": row["failure_type"],
        "matched_iou": float(row["matched_iou"]),
        "best_iou": float(row["best_iou"]),
        "matched_score": float(row["matched_score"]),
        "best_score": float(row["best_score"]),
        "gt_box": gt_box,
        "matched_box": matched_box,
        "best_box": best_box,
        "visualization": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize large crack IoU diagnosis cases.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--diagnosis-csv", required=True)
    parser.add_argument("--out-dir", default="outputs/visualizations/large_iou_cases")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-iou", type=float, default=None)
    parser.add_argument("--max-side", type=int, default=1600)
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    rows = load_rows(Path(args.diagnosis_csv), limit=args.limit, max_iou=args.max_iou)
    out_dir = Path(args.out_dir)
    index = []
    for rank, row in enumerate(rows, start=1):
        image_stem = Path(row["image"]).stem
        out_path = out_dir / f"{rank:02d}_id{row['image_id']}_{image_stem}_miou{float(row['matched_iou']):.3f}.jpg"
        rec = visualize_row(row, dataset_root=dataset_root, out_path=out_path, max_side=args.max_side)
        index.append(rec)
        print(f"Saved {out_path}")
    save_json(index, out_dir / "index.json")
    print(f"Saved {out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
