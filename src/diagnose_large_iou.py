from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from common import box_iou, clip_xyxy, load_json, save_json, xywh_to_xyxy
from validate_yolo import compute_matches


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
    raise ValueError(f"Unsupported split: {split}")


def load_gt_boxes(item: dict[str, Any], image_root: Path, eval_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    image_path = image_root / item["Image"]
    with Image.open(image_path) as im:
        width, height = im.width, im.height

    boxes = []
    for ann_idx, ann in enumerate(item.get("Annotations", [])):
        box = clip_xyxy(xywh_to_xyxy(ann["bbox"]), width, height)
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        area = box_w * box_h
        boxes.append(
            {
                "ann_idx": ann_idx,
                "box": box,
                "area": area,
                "width": box_w,
                "height": box_h,
                "large": area >= eval_cfg["large_area"],
                "matched": False,
                "matched_iou": 0.0,
                "matched_score": 0.0,
                "matched_box": None,
            }
        )
    return boxes


def load_pred_boxes(row: dict[str, Any]) -> list[dict[str, Any]]:
    preds = []
    for idx, box in enumerate(row.get("predict_bboxes", [])):
        preds.append(
            {
                "pred_idx": idx,
                "box": [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])],
                "score": float(box.get("score", 0.0)),
            }
        )
    return preds


def annotate_matches(
    gts_by_image: dict[int, list[dict[str, Any]]],
    preds_by_image: dict[int, list[dict[str, Any]]],
    iou_thr: float,
) -> None:
    pred_rows = []
    for image_id, preds in preds_by_image.items():
        for pred in preds:
            pred_rows.append((image_id, pred["score"], pred["box"], pred["pred_idx"]))
    pred_rows.sort(key=lambda item: item[1], reverse=True)

    for image_id, score, pred_box, pred_idx in pred_rows:
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gts_by_image[image_id]):
            if gt["matched"]:
                continue
            iou_val = box_iou(pred_box, gt["box"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_thr:
            gt = gts_by_image[image_id][best_idx]
            gt["matched"] = True
            gt["matched_iou"] = best_iou
            gt["matched_score"] = score
            gt["matched_box"] = pred_box
            gt["matched_pred_idx"] = pred_idx


def shape_metrics(gt_box: list[float], pred_box: list[float]) -> dict[str, float]:
    gx1, gy1, gx2, gy2 = gt_box
    px1, py1, px2, py2 = pred_box
    gw, gh = gx2 - gx1, gy2 - gy1
    pw, ph = px2 - px1, py2 - py1
    gcx, gcy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    inter_w = max(0.0, min(gx2, px2) - max(gx1, px1))
    inter_h = max(0.0, min(gy2, py2) - max(gy1, py1))
    inter = inter_w * inter_h
    gt_area = max(0.0, gw) * max(0.0, gh)
    pred_area = max(0.0, pw) * max(0.0, ph)
    return {
        "pred_w": pw,
        "pred_h": ph,
        "pred_area": pred_area,
        "w_ratio": pw / gw if gw else 0.0,
        "h_ratio": ph / gh if gh else 0.0,
        "area_ratio": pred_area / gt_area if gt_area else 0.0,
        "dx_norm": (pcx - gcx) / gw if gw else 0.0,
        "dy_norm": (pcy - gcy) / gh if gh else 0.0,
        "gt_coverage": inter / gt_area if gt_area else 0.0,
        "pred_precision_area": inter / pred_area if pred_area else 0.0,
    }


def summarize_failure(best_iou: float, matched_iou: float, best_score: float, matched_score: float) -> str:
    if best_iou < 0.5:
        return "no_good_candidate"
    if best_iou - matched_iou > 0.1 and best_score < matched_score:
        return "good_candidate_low_score"
    if matched_iou < 0.85 <= best_iou:
        return "good_candidate_not_matched"
    if matched_iou < 0.85:
        return "shape_quality_gap"
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose large-object IoU bottlenecks for a submission JSON.")
    parser.add_argument("--config", default="configs/yolo_seg_crack_hybrid.yaml")
    parser.add_argument("--submit", required=True)
    parser.add_argument("--split", choices=["val", "trainval"], default="val")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_root = Path(cfg["dataset_root"])
    prepared_root = Path(cfg["prepared_root"])
    eval_cfg = cfg["eval"]

    items = load_eval_items(dataset_root, prepared_root, args.split)
    rows_by_id = {int(row["ID"]): row for row in load_json(args.submit)}
    image_root = dataset_root / "trainval"

    gts_by_image = {item["ID"]: load_gt_boxes(item, image_root, eval_cfg) for item in items}
    preds_by_image = {
        item["ID"]: load_pred_boxes(rows_by_id.get(item["ID"], {"predict_bboxes": []}))
        for item in items
    }
    annotate_matches(gts_by_image, preds_by_image, float(eval_cfg["iou_match"]))

    records: list[dict[str, Any]] = []
    for item in items:
        image_id = item["ID"]
        preds = preds_by_image[image_id]
        for gt in gts_by_image[image_id]:
            if not gt["large"]:
                continue
            scored = [
                {
                    "pred_idx": pred["pred_idx"],
                    "score": pred["score"],
                    "iou": box_iou(pred["box"], gt["box"]),
                    "box": pred["box"],
                }
                for pred in preds
            ]
            best = max(scored, key=lambda row: row["iou"], default=None)
            by_score = sorted(scored, key=lambda row: row["score"], reverse=True)[: args.top_k]
            by_iou = sorted(scored, key=lambda row: row["iou"], reverse=True)[: args.top_k]

            best_box = best["box"] if best else [0.0, 0.0, 0.0, 0.0]
            matched_box = gt["matched_box"] or [0.0, 0.0, 0.0, 0.0]
            best_iou = best["iou"] if best else 0.0
            best_score = best["score"] if best else 0.0
            matched_iou = gt["matched_iou"]
            matched_score = gt["matched_score"]
            rec = {
                "image_id": image_id,
                "image": item["Image"],
                "ann_idx": gt["ann_idx"],
                "gt_area": gt["area"],
                "gt_w": gt["width"],
                "gt_h": gt["height"],
                "matched": gt["matched"],
                "matched_iou": matched_iou,
                "matched_score": matched_score,
                "best_iou": best_iou,
                "best_score": best_score,
                "iou_gap": best_iou - matched_iou,
                "failure_type": summarize_failure(best_iou, matched_iou, best_score, matched_score),
                "gt_box": json.dumps(gt["box"], ensure_ascii=False),
                "matched_box": json.dumps(matched_box, ensure_ascii=False),
                "best_box": json.dumps(best_box, ensure_ascii=False),
                "top_by_score": json.dumps([(round(v["iou"], 4), round(v["score"], 4)) for v in by_score], ensure_ascii=False),
                "top_by_iou": json.dumps([(round(v["iou"], 4), round(v["score"], 4)) for v in by_iou], ensure_ascii=False),
            }
            rec.update({f"best_{k}": v for k, v in shape_metrics(gt["box"], best_box).items()})
            rec.update({f"matched_{k}": v for k, v in shape_metrics(gt["box"], matched_box).items()})
            records.append(rec)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) if records else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    failure_counts: dict[str, int] = {}
    for rec in records:
        failure_counts[rec["failure_type"]] = failure_counts.get(rec["failure_type"], 0) + 1
    summary = {
        "submission": args.submit,
        "split": args.split,
        "large_count": len(records),
        "large_mean_matched_iou": sum(float(r["matched_iou"]) for r in records) / len(records) if records else 0.0,
        "large_mean_best_iou": sum(float(r["best_iou"]) for r in records) / len(records) if records else 0.0,
        "failure_counts": failure_counts,
        "worst_by_matched_iou": sorted(
            [
                {
                    "image_id": rec["image_id"],
                    "image": rec["image"],
                    "matched_iou": rec["matched_iou"],
                    "best_iou": rec["best_iou"],
                    "failure_type": rec["failure_type"],
                }
                for rec in records
            ],
            key=lambda row: row["matched_iou"],
        )[:10],
    }
    save_json(summary, Path(args.out_json))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
