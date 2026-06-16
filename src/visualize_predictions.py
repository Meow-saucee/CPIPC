from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from common import clip_xyxy, load_json, xywh_to_xyxy


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str) -> None:
    x, y = xy
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(bbox, fill=fill)
    draw.text((x, y), text, fill="white", font=font)


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: str, label: str, width: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    draw_label(draw, (x1, max(0, y1 - 12)), label, color)


def load_submission(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_json(path)
    return {int(row["ID"]): row for row in rows}


def visualize_trainval(dataset_root: Path, item: dict[str, Any], out_path: Path) -> None:
    image_path = dataset_root / "trainval" / item["Image"]
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for ann_idx, ann in enumerate(item.get("Annotations", []), start=1):
        box = clip_xyxy(xywh_to_xyxy(ann["bbox"]), width, height)
        draw_box(draw, box, "lime", f"GT {ann_idx}", width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def visualize_test(dataset_root: Path, item: dict[str, Any], submission_row: dict[str, Any] | None, out_path: Path) -> None:
    image_path = dataset_root / "test" / item["Image"]
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    if submission_row:
        for pred_idx, pred in enumerate(submission_row.get("predict_bboxes", []), start=1):
            box = clip_xyxy([pred["x1"], pred["y1"], pred["x2"], pred["y2"]], width, height)
            draw_box(draw, box, "red", f"P{pred_idx} {float(pred.get('score', 0.0)):.2f}", width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ground-truth boxes or submission predictions.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--split", choices=["trainval", "test"], default="trainval")
    parser.add_argument("--submit", default=None, help="Submission JSON for test visualization.")
    parser.add_argument("--out", default="outputs/visualizations")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--ids", default=None, help="Comma-separated image IDs to visualize.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_root = Path(args.out)
    ids = None
    if args.ids:
        ids = {int(value.strip()) for value in args.ids.split(",") if value.strip()}

    if args.split == "trainval":
        items = load_json(dataset_root / "trainval" / "trainval.json")["Dataset"]
        if ids is not None:
            items = [item for item in items if int(item["ID"]) in ids]
        for item in items[: args.limit]:
            out_path = out_root / "trainval" / f"{item['ID']}_{Path(item['Image']).stem}.jpg"
            visualize_trainval(dataset_root, item, out_path)
            print(f"Saved {out_path}")
        return

    items = load_json(dataset_root / "test" / "test.json")["Dataset"]
    if ids is not None:
        items = [item for item in items if int(item["ID"]) in ids]
    submission = load_submission(Path(args.submit)) if args.submit else {}
    for item in items[: args.limit]:
        out_path = out_root / "test" / f"{item['ID']}_{Path(item['Image']).stem}.jpg"
        visualize_test(dataset_root, item, submission.get(int(item["ID"])), out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
