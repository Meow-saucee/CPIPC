from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from common import box_iou, clip_xyxy, load_json, save_json


def to_box(pred: dict[str, Any]) -> list[float]:
    return [float(pred["x1"]), float(pred["y1"]), float(pred["x2"]), float(pred["y2"])]


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def merge_two_boxes(a: list[float], b: list[float], mode: str) -> list[float]:
    if mode == "union":
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
    if mode == "weighted":
        area_a = area(a)
        area_b = area(b)
        total = max(1e-6, area_a + area_b)
        return [(a[i] * area_a + b[i] * area_b) / total for i in range(4)]
    if mode == "larger":
        return a if area(a) >= area(b) else b
    if mode == "higher_score":
        return a
    raise ValueError(f"Unsupported merge mode: {mode}")


def merge_preds(preds: list[dict[str, Any]], iou_thr: float, mode: str, max_preds: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for pred in sorted(preds, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        pred_box = to_box(pred)
        merged = False
        for kept_pred in kept:
            kept_box = to_box(kept_pred)
            if box_iou(pred_box, kept_box) < iou_thr:
                continue
            if mode == "higher_score":
                kept_pred["score"] = max(float(kept_pred.get("score", 0.0)), float(pred.get("score", 0.0)))
            else:
                new_box = merge_two_boxes(kept_box, pred_box, mode)
                kept_pred["x1"], kept_pred["y1"], kept_pred["x2"], kept_pred["y2"] = new_box
                kept_pred["score"] = max(float(kept_pred.get("score", 0.0)), float(pred.get("score", 0.0)))
            merged = True
            break
        if not merged:
            kept.append(dict(pred))
    kept.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    if max_preds > 0:
        kept = kept[:max_preds]
    return kept


def load_image_sizes(dataset_root: Path, row_count: int) -> dict[int, tuple[int, int]]:
    sizes: dict[int, tuple[int, int]] = {}
    for split in ["test", "trainval"]:
        json_path = dataset_root / split / ("test.json" if split == "test" else "trainval.json")
        image_root = dataset_root / split
        if not json_path.exists():
            continue
        for item in load_json(json_path).get("Dataset", []):
            image_id = int(item["ID"])
            if image_id in sizes:
                continue
            image_path = image_root / item["Image"]
            if not image_path.exists():
                continue
            with Image.open(image_path) as im:
                sizes[image_id] = (im.width, im.height)
    if len(sizes) < row_count:
        print(f"Warning: only loaded {len(sizes)} image sizes for {row_count} rows.")
    return sizes


def clip_preds(row: dict[str, Any], size: tuple[int, int] | None) -> None:
    if size is None:
        return
    width, height = size
    clipped = []
    for pred in row.get("predict_bboxes", []):
        box = clip_xyxy(to_box(pred), width, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        item = dict(pred)
        item["x1"], item["y1"], item["x2"], item["y2"] = box
        clipped.append(item)
    row["predict_bboxes"] = clipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple official-format submission JSON files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iou-thr", type=float, default=0.65)
    parser.add_argument("--mode", choices=["higher_score", "union", "weighted", "larger"], default="higher_score")
    parser.add_argument("--score-scale", type=float, nargs="+", default=None, help="Optional per-input score multipliers.")
    parser.add_argument("--max-preds", type=int, default=300)
    parser.add_argument("--dataset", default="dataset", help="Dataset root used to clip merged boxes to image bounds.")
    parser.add_argument("--no-clip", action="store_true")
    args = parser.parse_args()

    submissions = [load_json(path) for path in args.inputs]
    if not submissions:
        raise ValueError("No submissions provided.")
    row_count = len(submissions[0])
    if any(len(sub) != row_count for sub in submissions):
        raise ValueError("All submissions must have the same number of rows.")
    scales = args.score_scale or [1.0] * len(submissions)
    if len(scales) != len(submissions):
        raise ValueError("--score-scale length must match --inputs length.")
    image_sizes = {} if args.no_clip else load_image_sizes(Path(args.dataset), row_count)

    output_rows: list[dict[str, Any]] = []
    for row_idx in range(row_count):
        base = dict(submissions[0][row_idx])
        image_id = int(base["ID"])
        preds: list[dict[str, Any]] = []
        for sub, scale in zip(submissions, scales):
            row = sub[row_idx]
            if int(row["ID"]) != image_id:
                raise ValueError(f"Row ID mismatch at index {row_idx}: {image_id} != {row['ID']}")
            for pred in row.get("predict_bboxes", []):
                item = dict(pred)
                item["score"] = min(1.0, max(0.0, float(item.get("score", 0.0)) * float(scale)))
                item["label"] = item.get("label", "crack")
                preds.append(item)
        base["predict_bboxes"] = merge_preds(preds, args.iou_thr, args.mode, args.max_preds)
        clip_preds(base, image_sizes.get(image_id))
        output_rows.append(base)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_rows, out_path)
    print(f"Saved merged submission to {out_path}")
    print(f"rows={len(output_rows)}, preds={sum(len(row.get('predict_bboxes', [])) for row in output_rows)}")


if __name__ == "__main__":
    main()
