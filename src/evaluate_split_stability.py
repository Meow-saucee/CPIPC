from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from common import load_json, save_json
from eval_submission import gt_boxes, pred_boxes, update_best_ious
from prepare_yolo import image_flags, stratified_split
from validate_yolo import ap_from_pr, compute_matches


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.replace("\n", ",").split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def annotation_scale_counts(items: list[dict[str, Any]], eval_cfg: dict[str, Any]) -> dict[str, int]:
    tiny_boxes = 0
    large_boxes = 0
    total_boxes = 0
    tiny_images = 0
    large_images = 0
    for item in items:
        has_tiny, has_large = image_flags(item)
        tiny_images += int(has_tiny)
        large_images += int(has_large)
        for ann in item.get("Annotations", []):
            _, _, w, h = map(float, ann["bbox"])
            area = w * h
            total_boxes += 1
            tiny_boxes += int(w <= eval_cfg["tiny_width"] or area <= eval_cfg["tiny_area"])
            large_boxes += int(area >= eval_cfg["large_area"])
    return {
        "gt_boxes": total_boxes,
        "tiny_images": tiny_images,
        "large_images": large_images,
        "tiny_boxes": tiny_boxes,
        "large_boxes": large_boxes,
    }


def evaluate_items(
    items: list[dict[str, Any]],
    rows_by_id: dict[int, dict[str, Any]],
    dataset_root: Path,
    eval_cfg: dict[str, Any],
    allow_missing_as_empty: bool,
) -> dict[str, Any]:
    missing_ids = [int(item["ID"]) for item in items if int(item["ID"]) not in rows_by_id]
    if missing_ids and not allow_missing_as_empty:
        return {
            "evaluable": False,
            "missing_predictions": len(missing_ids),
            "missing_ids_preview": missing_ids[:20],
            "reason": "submission does not contain predictions for all images in this split",
        }

    gts_by_image = {int(item["ID"]): gt_boxes(item, dataset_root, eval_cfg) for item in items}
    preds_by_image = {
        int(item["ID"]): pred_boxes(rows_by_id.get(int(item["ID"]), {"predict_bboxes": []}))
        for item in items
    }
    update_best_ious(gts_by_image, preds_by_image)
    tp, fp, total_gt = compute_matches(gts_by_image, preds_by_image, eval_cfg["iou_match"])
    matched = sum(tp)
    pred_count = len(tp)
    tiny_gts = [gt for gts in gts_by_image.values() for gt in gts if gt["tiny"]]
    large_gts = [gt for gts in gts_by_image.values() for gt in gts if gt["large"]]
    return {
        "evaluable": True,
        "missing_predictions": len(missing_ids),
        "predicted_boxes": pred_count,
        "precision_at_conf": matched / pred_count if pred_count else 0.0,
        "recall_at_iou50": matched / total_gt if total_gt else 0.0,
        "mAP50": ap_from_pr(tp, fp, total_gt),
        "tiny_recall_at_iou50": sum(1 for gt in tiny_gts if gt["matched"]) / len(tiny_gts) if tiny_gts else None,
        "large_mean_matched_iou": sum(gt["matched_iou"] for gt in large_gts) / len(large_gts) if large_gts else None,
        "large_mean_best_iou": sum(gt["best_iou"] for gt in large_gts) / len(large_gts) if large_gts else None,
    }


def summarize_metric(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(row[key]) for row in rows if row.get("evaluable") and row.get(key) is not None]
    if not values:
        return {"min": None, "max": None, "mean": None, "range": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "range": max(values) - min(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit alternative stratified train/val splits and, when a full-trainval "
            "prediction JSON is provided, evaluate metric stability across those splits."
        )
    )
    parser.add_argument("--config", default="configs/yolo_seg_crack_hybrid.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seeds", default="0,1,2,3,4,42")
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--submission", default=None, help="Submission-style JSON that should cover all trainval images.")
    parser.add_argument("--allow-missing-as-empty", action="store_true")
    parser.add_argument("--out-json", default="outputs/reports/split_stability_summary.json")
    parser.add_argument("--out-csv", default="outputs/reports/split_stability_summary.csv")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_root = Path(args.dataset or cfg["dataset_root"])
    eval_cfg = cfg["eval"]
    split_cfg = cfg.get("split", {})
    val_ratio = args.val_ratio if args.val_ratio is not None else float(split_cfg.get("val_ratio", 0.2))
    seeds = parse_seeds(args.seeds)
    items = load_json(dataset_root / "trainval" / "trainval.json")["Dataset"]

    rows_by_id: dict[int, dict[str, Any]] | None = None
    if args.submission:
        submission_rows = load_json(args.submission)
        rows_by_id = {int(row["ID"]): row for row in submission_rows}

    _, base_val_items = stratified_split(items, val_ratio=val_ratio, seed=int(split_cfg.get("seed", 42)))
    base_val_ids = {int(item["ID"]) for item in base_val_items}

    split_rows: list[dict[str, Any]] = []
    for seed in seeds:
        train_items, val_items = stratified_split(items, val_ratio=val_ratio, seed=seed)
        val_ids = {int(item["ID"]) for item in val_items}
        stats = annotation_scale_counts(val_items, eval_cfg)
        row: dict[str, Any] = {
            "seed": seed,
            "val_ratio": val_ratio,
            "train_images": len(train_items),
            "val_images": len(val_items),
            "val_overlap_with_config_seed42": len(val_ids & base_val_ids),
            "val_jaccard_with_config_seed42": len(val_ids & base_val_ids) / len(val_ids | base_val_ids),
            **stats,
        }
        if rows_by_id is not None:
            row.update(
                evaluate_items(
                    val_items,
                    rows_by_id=rows_by_id,
                    dataset_root=dataset_root,
                    eval_cfg=eval_cfg,
                    allow_missing_as_empty=args.allow_missing_as_empty,
                )
            )
        else:
            row.update({"evaluable": False, "reason": "no submission provided"})
        split_rows.append(row)

    metric_summary = {
        key: summarize_metric(split_rows, key)
        for key in [
            "mAP50",
            "recall_at_iou50",
            "tiny_recall_at_iou50",
            "large_mean_matched_iou",
            "large_mean_best_iou",
        ]
    }
    report = {
        "config": args.config,
        "dataset": str(dataset_root),
        "submission": args.submission,
        "seeds": seeds,
        "val_ratio": val_ratio,
        "note": (
            "Alternative split metrics are meaningful only if the predictions were generated by models "
            "that did not train on the evaluated val images. A single seed=42 trained model evaluated "
            "on new splits can include train leakage."
        ),
        "splits": split_rows,
        "metric_summary": metric_summary,
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    save_json(report, out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in split_rows for key in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)

    print(f"Saved split stability JSON to {out_json}")
    print(f"Saved split stability CSV to {out_csv}")
    for row in split_rows:
        status = "evaluable" if row.get("evaluable") else f"not_evaluable: {row.get('reason')}"
        print(
            f"seed={row['seed']} val={row['val_images']} gt={row['gt_boxes']} "
            f"tiny={row['tiny_boxes']} large={row['large_boxes']} {status}"
        )


if __name__ == "__main__":
    main()
