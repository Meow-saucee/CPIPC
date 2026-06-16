from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from common import box_iou, clip_xyxy, load_json, save_json, xywh_to_xyxy
from eval_submission import gt_boxes, load_eval_items, pred_boxes, update_best_ious
from validate_yolo import ap_from_pr, compute_matches


@dataclass(frozen=True)
class SearchParams:
    name: str
    expand_x: float
    expand_y: float
    shrink_x: float
    shrink_y: float
    shift_x: float
    shift_y: float
    score_factor: float
    score_floor: float
    max_new_per_image: int
    min_area: float
    min_side: float
    only_large_or_long: bool


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def transform_box(
    box: list[float],
    width: int,
    height: int,
    expand_x: float,
    expand_y: float,
    shrink_x: float,
    shrink_y: float,
    shift_x: float,
    shift_y: float,
) -> list[float]:
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0 + shift_x * w
    cy = (y1 + y2) / 2.0 + shift_y * h
    new_w = max(1.0, w * (1.0 + expand_x - shrink_x))
    new_h = max(1.0, h * (1.0 + expand_y - shrink_y))
    return clip_xyxy([cx - new_w / 2.0, cy - new_h / 2.0, cx + new_w / 2.0, cy + new_h / 2.0], width, height)


def is_large_or_long(box: list[float], min_area: float, min_side: float) -> bool:
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    if w * h >= min_area:
        return True
    return max(w, h) >= min_side and max(w, h) / max(1.0, min(w, h)) >= 3.0


def add_variants(rows: list[dict[str, Any]], image_sizes: dict[int, tuple[int, int]], params: SearchParams) -> list[dict[str, Any]]:
    out_rows = copy.deepcopy(rows)
    for row in out_rows:
        image_id = int(row["ID"])
        width, height = image_sizes[image_id]
        preds = row.get("predict_bboxes", [])
        candidates: list[dict[str, Any]] = []
        for pred in preds:
            box = [float(pred["x1"]), float(pred["y1"]), float(pred["x2"]), float(pred["y2"])]
            if params.only_large_or_long and not is_large_or_long(box, params.min_area, params.min_side):
                continue
            variant = transform_box(
                box,
                width,
                height,
                expand_x=params.expand_x,
                expand_y=params.expand_y,
                shrink_x=params.shrink_x,
                shrink_y=params.shrink_y,
                shift_x=params.shift_x,
                shift_y=params.shift_y,
            )
            if box_iou(box, variant) > 0.995:
                continue
            score = max(float(pred.get("score", 0.0)) * params.score_factor, params.score_floor)
            candidates.append(
                {
                    "x1": variant[0],
                    "y1": variant[1],
                    "x2": variant[2],
                    "y2": variant[3],
                    "score": min(1.0, score),
                    "label": pred.get("label", "crack"),
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        preds.extend(candidates[: params.max_new_per_image])
        preds.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        row["predict_bboxes"] = preds
    return out_rows


def evaluate_rows(
    rows: list[dict[str, Any]],
    items: list[dict[str, Any]],
    dataset_root: Path,
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    rows_by_id = {int(row["ID"]): row for row in rows}
    gts_by_image = {item["ID"]: gt_boxes(item, dataset_root, eval_cfg) for item in items}
    preds_by_image = {item["ID"]: pred_boxes(rows_by_id.get(item["ID"], {"predict_bboxes": []})) for item in items}
    update_best_ious(gts_by_image, preds_by_image)
    tp, fp, total_gt = compute_matches(gts_by_image, preds_by_image, eval_cfg["iou_match"])
    matched = sum(tp)
    pred_count = len(tp)
    tiny_gts = [gt for gts in gts_by_image.values() for gt in gts if gt["tiny"]]
    large_gts = [gt for gts in gts_by_image.values() for gt in gts if gt["large"]]
    return {
        "images": len(items),
        "ground_truth_boxes": total_gt,
        "predicted_boxes": pred_count,
        "precision_at_conf": matched / pred_count if pred_count else 0.0,
        "recall_at_iou50": matched / total_gt if total_gt else 0.0,
        "mAP50": ap_from_pr(tp, fp, total_gt),
        "tiny_recall_at_iou50": sum(1 for gt in tiny_gts if gt["matched"]) / len(tiny_gts) if tiny_gts else None,
        "large_mean_matched_iou": sum(gt["matched_iou"] for gt in large_gts) / len(large_gts) if large_gts else None,
        "large_mean_best_iou": sum(gt["best_iou"] for gt in large_gts) / len(large_gts) if large_gts else None,
    }


def build_search_space(args: argparse.Namespace) -> list[SearchParams]:
    variants: list[SearchParams] = []
    idx = 0
    for expand_x in args.expand_x:
        for expand_y in args.expand_y:
            for shrink_x in args.shrink_x:
                for shrink_y in args.shrink_y:
                    for shift_x in args.shift_x:
                        for shift_y in args.shift_y:
                            for score_factor in args.score_factor:
                                for score_floor in args.score_floor:
                                    idx += 1
                                    variants.append(
                                        SearchParams(
                                            name=f"largepp_{idx:04d}",
                                            expand_x=expand_x,
                                            expand_y=expand_y,
                                            shrink_x=shrink_x,
                                            shrink_y=shrink_y,
                                            shift_x=shift_x,
                                            shift_y=shift_y,
                                            score_factor=score_factor,
                                            score_floor=score_floor,
                                            max_new_per_image=args.max_new_per_image,
                                            min_area=args.min_area,
                                            min_side=args.min_side,
                                            only_large_or_long=not args.all_boxes,
                                        )
                                    )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline search for large-box post-processing on a validation submission JSON.")
    parser.add_argument("--config", default="configs/yolo_seg_crack_hybrid.yaml")
    parser.add_argument("--submit", required=True)
    parser.add_argument("--split", choices=["val", "trainval"], default="val")
    parser.add_argument("--out-dir", default="outputs/large_box_search")
    parser.add_argument("--top-k-save", type=int, default=5)
    parser.add_argument("--max-new-per-image", type=int, default=8)
    parser.add_argument("--min-area", type=float, default=90000.0)
    parser.add_argument("--min-side", type=float, default=900.0)
    parser.add_argument("--all-boxes", action="store_true")
    parser.add_argument("--expand-x", type=float, nargs="+", default=[0.0, 0.08, 0.16])
    parser.add_argument("--expand-y", type=float, nargs="+", default=[0.0, 0.08, 0.16])
    parser.add_argument("--shrink-x", type=float, nargs="+", default=[0.0, 0.08, 0.16])
    parser.add_argument("--shrink-y", type=float, nargs="+", default=[0.0, 0.08, 0.16])
    parser.add_argument("--shift-x", type=float, nargs="+", default=[0.0])
    parser.add_argument("--shift-y", type=float, nargs="+", default=[0.0])
    parser.add_argument("--score-factor", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--score-floor", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_root = Path(cfg["dataset_root"])
    prepared_root = Path(cfg["prepared_root"])
    eval_cfg = cfg["eval"]
    items = load_eval_items(dataset_root, prepared_root, args.split)
    image_root = dataset_root / "trainval"
    image_sizes: dict[int, tuple[int, int]] = {}
    for item in items:
        with Image.open(image_root / item["Image"]) as im:
            image_sizes[int(item["ID"])] = (im.width, im.height)

    rows = load_json(args.submit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_metrics = evaluate_rows(rows, items, dataset_root, eval_cfg)
    results: list[dict[str, Any]] = [{"name": "base", **base_metrics}]
    candidates = build_search_space(args)
    for params in candidates:
        variant_rows = add_variants(rows, image_sizes, params)
        metrics = evaluate_rows(variant_rows, items, dataset_root, eval_cfg)
        results.append(
            {
                "name": params.name,
                "expand_x": params.expand_x,
                "expand_y": params.expand_y,
                "shrink_x": params.shrink_x,
                "shrink_y": params.shrink_y,
                "shift_x": params.shift_x,
                "shift_y": params.shift_y,
                "score_factor": params.score_factor,
                "score_floor": params.score_floor,
                "max_new_per_image": params.max_new_per_image,
                "min_area": params.min_area,
                "min_side": params.min_side,
                "only_large_or_long": params.only_large_or_long,
                **metrics,
            }
        )

    summary_path = out_dir / "summary.csv"
    fieldnames = list(results[0].keys())
    for row in results[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    ranked = sorted(
        results[1:],
        key=lambda row: (
            float(row["large_mean_matched_iou"] or 0.0),
            float(row["mAP50"] or 0.0),
            float(row["recall_at_iou50"] or 0.0),
        ),
        reverse=True,
    )
    save_json({"base": base_metrics, "top": ranked[:20]}, out_dir / "top_results.json")

    for row in ranked[: args.top_k_save]:
        params = SearchParams(
            name=str(row["name"]),
            expand_x=float(row["expand_x"]),
            expand_y=float(row["expand_y"]),
            shrink_x=float(row["shrink_x"]),
            shrink_y=float(row["shrink_y"]),
            shift_x=float(row["shift_x"]),
            shift_y=float(row["shift_y"]),
            score_factor=float(row["score_factor"]),
            score_floor=float(row["score_floor"]),
            max_new_per_image=int(row["max_new_per_image"]),
            min_area=float(row["min_area"]),
            min_side=float(row["min_side"]),
            only_large_or_long=bool(row["only_large_or_long"]),
        )
        save_json(add_variants(rows, image_sizes, params), out_dir / f"{params.name}.json")

    print(f"Saved search summary to {summary_path}")
    print(json.dumps({"base": base_metrics, "top5": ranked[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
