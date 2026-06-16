from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from typing import Any

from common import box_iou, load_json, save_json


def to_box(pred: dict[str, Any]) -> list[float]:
    return [float(pred["x1"]), float(pred["y1"]), float(pred["x2"]), float(pred["y2"])]


def set_box(pred: dict[str, Any], box: list[float]) -> None:
    pred["x1"], pred["y1"], pred["x2"], pred["y2"] = box


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_aspect(box: list[float]) -> float:
    width = max(1e-6, box[2] - box[0])
    height = max(1e-6, box[3] - box[1])
    return max(width, height) / max(1e-6, min(width, height))


def is_large_or_long(box: list[float], min_area: float, min_side: float, min_aspect: float) -> bool:
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    if width * height >= min_area:
        return True
    return max(width, height) >= min_side and box_aspect(box) >= min_aspect


def find(parent: list[int], idx: int) -> int:
    while parent[idx] != idx:
        parent[idx] = parent[parent[idx]]
        idx = parent[idx]
    return idx


def unite(parent: list[int], a: int, b: int) -> None:
    ra = find(parent, a)
    rb = find(parent, b)
    if ra != rb:
        parent[rb] = ra


def clustered_indices(boxes: list[list[float]], iou_thr: float) -> list[list[int]]:
    parent = list(range(len(boxes)))
    for i, box_i in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            if box_iou(box_i, boxes[j]) >= iou_thr:
                unite(parent, i, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(parent, i), []).append(i)
    return list(groups.values())


def consensus_score(idx: int, group: list[int], boxes: list[list[float]], scores: list[float]) -> float:
    if len(group) <= 1:
        return 0.0
    total = 0.0
    denom = 0.0
    for other in group:
        if other == idx:
            continue
        weight = max(0.05, scores[other])
        total += box_iou(boxes[idx], boxes[other]) * weight
        denom += weight
    return total / max(1e-6, denom)


def area_quality(box: list[float], target_area: float) -> float:
    current = max(1.0, box_area(box))
    target = max(1.0, target_area)
    ratio = max(current / target, target / current)
    return 1.0 / ratio


def rescore_row(
    row: dict[str, Any],
    min_area: float,
    min_side: float,
    min_aspect: float,
    cluster_iou: float,
    score_floor: float,
    score_ceiling: float,
    original_weight: float,
    consensus_weight: float,
    area_weight: float,
    demote_oversized: float,
    oversized_ratio: float,
) -> None:
    preds = row.get("predict_bboxes", [])
    boxes = [to_box(pred) for pred in preds]
    selected = [
        idx
        for idx, box in enumerate(boxes)
        if is_large_or_long(box, min_area=min_area, min_side=min_side, min_aspect=min_aspect)
    ]
    if not selected:
        return

    selected_boxes = [boxes[idx] for idx in selected]
    scores = [float(preds[idx].get("score", 0.0)) for idx in selected]
    for group_local in clustered_indices(selected_boxes, cluster_iou):
        if len(group_local) < 2:
            continue
        group_global = [selected[idx] for idx in group_local]
        group_boxes = [boxes[idx] for idx in group_global]
        group_scores = [float(preds[idx].get("score", 0.0)) for idx in group_global]
        areas = [box_area(box) for box in group_boxes]
        target_area = median(areas)
        for local_idx, global_idx in enumerate(group_global):
            original = float(preds[global_idx].get("score", 0.0))
            consensus = consensus_score(local_idx, list(range(len(group_boxes))), group_boxes, group_scores)
            quality = area_quality(group_boxes[local_idx], target_area)
            new_score = (
                original_weight * original
                + consensus_weight * consensus
                + area_weight * quality
            )
            if box_area(group_boxes[local_idx]) > target_area * oversized_ratio:
                new_score *= demote_oversized
            preds[global_idx]["score"] = min(score_ceiling, max(score_floor, new_score))

    preds.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    row["predict_bboxes"] = preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore large/elongated box clusters by geometric consistency.")
    parser.add_argument("--submit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-area", type=float, default=60000.0)
    parser.add_argument("--min-side", type=float, default=700.0)
    parser.add_argument("--min-aspect", type=float, default=3.0)
    parser.add_argument("--cluster-iou", type=float, default=0.25)
    parser.add_argument("--score-floor", type=float, default=0.05)
    parser.add_argument("--score-ceiling", type=float, default=0.99)
    parser.add_argument("--original-weight", type=float, default=0.30)
    parser.add_argument("--consensus-weight", type=float, default=0.45)
    parser.add_argument("--area-weight", type=float, default=0.25)
    parser.add_argument("--demote-oversized", type=float, default=0.75)
    parser.add_argument("--oversized-ratio", type=float, default=1.45)
    args = parser.parse_args()

    rows = load_json(args.submit)
    for row in rows:
        rescore_row(
            row,
            min_area=args.min_area,
            min_side=args.min_side,
            min_aspect=args.min_aspect,
            cluster_iou=args.cluster_iou,
            score_floor=args.score_floor,
            score_ceiling=args.score_ceiling,
            original_weight=args.original_weight,
            consensus_weight=args.consensus_weight,
            area_weight=args.area_weight,
            demote_oversized=args.demote_oversized,
            oversized_ratio=args.oversized_ratio,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(rows, out_path)
    print(f"Saved cluster-rescored submission to {out_path}")
    print(f"rows={len(rows)}, preds={sum(len(row.get('predict_bboxes', [])) for row in rows)}")


if __name__ == "__main__":
    main()
