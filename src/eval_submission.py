from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from common import box_iou, clip_xyxy, load_json, save_json, xywh_to_xyxy
from validate_yolo import ap_from_pr, compute_matches


def load_eval_items(dataset_root: Path, prepared_root: Path, split: str) -> list[dict[str, Any]]:
    if split == "val":
        all_items = load_json(dataset_root / "trainval" / "trainval.json")["Dataset"]
        manifest_path = prepared_root / "split_manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} not found.")
        with manifest_path.open("r", encoding="utf-8") as f:
            val_ids = set(yaml.safe_load(f)["val_ids"])
        return [item for item in all_items if item["ID"] in val_ids]
    if split == "trainval":
        return load_json(dataset_root / "trainval" / "trainval.json")["Dataset"]
    raise ValueError(f"Unsupported split for GT evaluation: {split}")


def gt_boxes(item: dict[str, Any], dataset_root: Path, eval_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    image_path = dataset_root / "trainval" / item["Image"]
    with Image.open(image_path) as im:
        width, height = im.width, im.height
    boxes = []
    for ann in item.get("Annotations", []):
        x1, y1, x2, y2 = clip_xyxy(xywh_to_xyxy(ann["bbox"]), width, height)
        w, h = x2 - x1, y2 - y1
        area = w * h
        boxes.append(
            {
                "box": [x1, y1, x2, y2],
                "area": area,
                "width": w,
                "tiny": w <= eval_cfg["tiny_width"] or area <= eval_cfg["tiny_area"],
                "large": area >= eval_cfg["large_area"],
                "matched": False,
                "matched_iou": 0.0,
                "best_iou": 0.0,
            }
        )
    return boxes


def pred_boxes(row: dict[str, Any]) -> list[dict[str, Any]]:
    preds = []
    for box in row.get("predict_bboxes", []):
        preds.append(
            {
                "box": [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])],
                "score": float(box.get("score", 0.0)),
            }
        )
    return preds


def update_best_ious(gts_by_image: dict[int, list[dict[str, Any]]], preds_by_image: dict[int, list[dict[str, Any]]]) -> None:
    for image_id, gts in gts_by_image.items():
        preds = preds_by_image.get(image_id, [])
        for gt in gts:
            gt["best_iou"] = max((box_iou(pred["box"], gt["box"]) for pred in preds), default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a submission-style JSON against trainval/val ground truth.")
    parser.add_argument("--config", default="configs/yolo_crack.yaml")
    parser.add_argument("--submit", required=True)
    parser.add_argument("--split", choices=["val", "trainval"], default="val")
    parser.add_argument("--out", default=None)
    parser.add_argument("--errors", default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_root = Path(cfg["dataset_root"])
    prepared_root = Path(cfg["prepared_root"])
    eval_cfg = cfg["eval"]
    reports_dir = Path(cfg["outputs"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    items = load_eval_items(dataset_root, prepared_root, args.split)
    rows = load_json(args.submit)
    rows_by_id = {int(row["ID"]): row for row in rows}

    gts_by_image = {item["ID"]: gt_boxes(item, dataset_root, eval_cfg) for item in items}
    preds_by_image = {item["ID"]: pred_boxes(rows_by_id.get(item["ID"], {"predict_bboxes": []})) for item in items}
    update_best_ious(gts_by_image, preds_by_image)

    tp, fp, total_gt = compute_matches(gts_by_image, preds_by_image, eval_cfg["iou_match"])
    matched = sum(tp)
    pred_count = len(tp)
    tiny_gts = [gt for gts in gts_by_image.values() for gt in gts if gt["tiny"]]
    large_gts = [gt for gts in gts_by_image.values() for gt in gts if gt["large"]]
    metrics = {
        "submission": args.submit,
        "split": args.split,
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

    error_path = Path(args.errors) if args.errors else reports_dir / f"submission_errors_{args.split}.csv"
    with error_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "image_id", "image", "iou", "box", "scale_tag"])
        writer.writeheader()
        for item in items:
            for gt in gts_by_image[item["ID"]]:
                if not gt["matched"]:
                    writer.writerow(
                        {
                            "kind": "FN",
                            "image_id": item["ID"],
                            "image": item["Image"],
                            "iou": gt["best_iou"],
                            "box": gt["box"],
                            "scale_tag": "tiny" if gt["tiny"] else "large" if gt["large"] else "normal",
                        }
                    )

    out_path = Path(args.out) if args.out else reports_dir / f"submission_metrics_{args.split}.json"
    save_json(metrics, out_path)
    print(f"Saved metrics to {out_path}")
    print(f"Saved errors to {error_path}")
    print(metrics)


if __name__ == "__main__":
    main()
