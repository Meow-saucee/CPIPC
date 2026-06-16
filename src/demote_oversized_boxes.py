from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import box_iou, load_json, save_json


def to_box(pred: dict[str, Any]) -> list[float]:
    return [float(pred["x1"]), float(pred["y1"]), float(pred["x2"]), float(pred["y2"])]


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Demote oversized boxes when a tighter overlapping candidate exists.")
    parser.add_argument("--submit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-area", type=float, default=90000.0)
    parser.add_argument("--contain-iou", type=float, default=0.45, help="IoU threshold between oversized and tighter candidate.")
    parser.add_argument("--area-ratio", type=float, default=1.45, help="Demote box if it is this many times larger than competitor.")
    parser.add_argument("--competitor-score-min", type=float, default=0.45)
    parser.add_argument("--demote-factor", type=float, default=0.45)
    parser.add_argument("--demote-below", type=float, default=0.49)
    args = parser.parse_args()

    rows = load_json(args.submit)
    changed = 0
    for row in rows:
        preds = row.get("predict_bboxes", [])
        boxes = [to_box(pred) for pred in preds]
        areas = [area(box) for box in boxes]
        for i, pred in enumerate(preds):
            if areas[i] < args.min_area:
                continue
            score = float(pred.get("score", 0.0))
            should_demote = False
            for j, other in enumerate(preds):
                if i == j:
                    continue
                other_score = float(other.get("score", 0.0))
                if other_score < args.competitor_score_min:
                    continue
                if areas[j] <= 0 or areas[i] / areas[j] < args.area_ratio:
                    continue
                if box_iou(boxes[i], boxes[j]) >= args.contain_iou:
                    should_demote = True
                    break
            if should_demote:
                pred["score"] = min(score * args.demote_factor, args.demote_below)
                changed += 1
        row["predict_bboxes"] = sorted(preds, key=lambda item: float(item.get("score", 0.0)), reverse=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(rows, out_path)
    print(f"Saved demoted submission to {out_path}")
    print(f"rows={len(rows)}, changed_boxes={changed}")


if __name__ == "__main__":
    main()
