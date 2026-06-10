from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from common import box_iou, clip_xyxy, load_json, load_yaml, save_json, xywh_to_xyxy


def load_val_items(dataset_root: Path, prepared_root: Path) -> list[dict[str, Any]]:
    all_items = load_json(dataset_root / "trainval" / "trainval.json")["Dataset"]
    manifest_path = prepared_root / "split_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found. Run src/prepare_yolo.py first.")
    with manifest_path.open("r", encoding="utf-8") as f:
        val_ids = set(yaml.safe_load(f)["val_ids"])
    return [item for item in all_items if item["ID"] in val_ids]


def gt_boxes(item: dict[str, Any], dataset_root: Path) -> list[dict[str, Any]]:
    image_path = dataset_root / "trainval" / item["Image"]
    with Image.open(image_path) as im:
        width, height = im.width, im.height
    boxes = []
    for ann in item.get("Annotations", []):
        x1, y1, x2, y2 = clip_xyxy(xywh_to_xyxy(ann["bbox"]), width, height)
        w, h = x2 - x1, y2 - y1
        boxes.append(
            {
                "box": [x1, y1, x2, y2],
                "area": w * h,
                "width": w,
                "tiny": w <= 5 or w * h <= 50,
                "large": w * h >= 90000,
                "matched": False,
                "matched_iou": 0.0,
            }
        )
    return boxes


def predict_image(model: Any, image_path: Path, imgsz: int, conf: float, iou: float, max_det: int) -> tuple[list[dict[str, Any]], float]:
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    start = time.perf_counter()
    results = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    preds: list[dict[str, Any]] = []
    result = results[0]
    if result.boxes is None:
        return preds, elapsed_ms
    for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
        preds.append({"box": [float(v) for v in box], "score": float(score)})
    return preds, elapsed_ms


def compute_matches(
    gts_by_image: dict[int, list[dict[str, Any]]],
    preds_by_image: dict[int, list[dict[str, Any]]],
    iou_thr: float,
) -> tuple[list[int], list[int], int]:
    pred_rows = []
    for image_id, preds in preds_by_image.items():
        for pred in preds:
            pred_rows.append((image_id, pred["score"], pred["box"]))
    pred_rows.sort(key=lambda x: x[1], reverse=True)
    tp: list[int] = []
    fp: list[int] = []
    total_gt = sum(len(v) for v in gts_by_image.values())

    for image_id, _score, pred_box in pred_rows:
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
            tp.append(1)
            fp.append(0)
            gt = gts_by_image[image_id][best_idx]
            gt["matched"] = True
            gt["matched_iou"] = best_iou
        else:
            tp.append(0)
            fp.append(1)
    return tp, fp, total_gt


def ap_from_pr(tp: list[int], fp: list[int], total_gt: int) -> float:
    if total_gt == 0:
        return 0.0
    cum_tp = []
    cum_fp = []
    t = f = 0
    for is_tp, is_fp in zip(tp, fp):
        t += is_tp
        f += is_fp
        cum_tp.append(t)
        cum_fp.append(f)
    recalls = [v / total_gt for v in cum_tp]
    precisions = [cum_tp[i] / max(1, cum_tp[i] + cum_fp[i]) for i in range(len(cum_tp))]
    ap = 0.0
    for threshold in [i / 100 for i in range(0, 101)]:
        p = max((precisions[i] for i, r in enumerate(recalls) if r >= threshold), default=0.0)
        ap += p / 101.0
    return ap


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO predictions with bbox AP50 and scale metrics.")
    parser.add_argument("--config", default="configs/yolo_crack.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_root = Path(cfg["dataset_root"])
    prepared_root = Path(cfg["prepared_root"])
    reports_dir = Path(cfg["outputs"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = cfg["eval"]
    infer_cfg = cfg["infer"]

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing ultralytics. Install dependencies from requirements.txt first.") from exc

    model = YOLO(args.weights)
    val_items = load_val_items(dataset_root, prepared_root)
    gts_by_image: dict[int, list[dict[str, Any]]] = {}
    preds_by_image: dict[int, list[dict[str, Any]]] = {}
    times: list[float] = []

    for item in val_items:
        image_path = dataset_root / "trainval" / item["Image"]
        gts_by_image[item["ID"]] = gt_boxes(item, dataset_root)
        preds, elapsed_ms = predict_image(
            model,
            image_path,
            imgsz=infer_cfg["imgsz"],
            conf=infer_cfg["conf"],
            iou=infer_cfg["iou"],
            max_det=infer_cfg["max_det"],
        )
        preds_by_image[item["ID"]] = preds
        times.append(elapsed_ms)

    tp, fp, total_gt = compute_matches(gts_by_image, preds_by_image, eval_cfg["iou_match"])
    matched = sum(tp)
    pred_count = len(tp)
    tiny_gts = [gt for gts in gts_by_image.values() for gt in gts if gt["tiny"]]
    large_gts = [gt for gts in gts_by_image.values() for gt in gts if gt["large"]]
    metrics = {
        "images": len(val_items),
        "ground_truth_boxes": total_gt,
        "predicted_boxes": pred_count,
        "precision_at_conf": matched / pred_count if pred_count else 0.0,
        "recall_at_iou50": matched / total_gt if total_gt else 0.0,
        "mAP50": ap_from_pr(tp, fp, total_gt),
        "tiny_recall_at_iou50": sum(1 for gt in tiny_gts if gt["matched"]) / len(tiny_gts) if tiny_gts else None,
        "large_mean_best_iou": sum(gt["matched_iou"] for gt in large_gts) / len(large_gts) if large_gts else None,
        "avg_inference_time_ms": sum(times) / len(times) if times else 0.0,
    }

    error_path = reports_dir / "val_errors.csv"
    with error_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "image_id", "image", "score", "iou", "box", "scale_tag"])
        writer.writeheader()
        for item in val_items:
            for gt in gts_by_image[item["ID"]]:
                if not gt["matched"]:
                    writer.writerow(
                        {
                            "kind": "FN",
                            "image_id": item["ID"],
                            "image": item["Image"],
                            "score": "",
                            "iou": gt["matched_iou"],
                            "box": gt["box"],
                            "scale_tag": "tiny" if gt["tiny"] else "large" if gt["large"] else "normal",
                        }
                    )

    out_path = Path(args.out) if args.out else reports_dir / "val_metrics.json"
    save_json(metrics, out_path)
    print(f"Saved metrics to {out_path}")
    print(f"Saved error analysis to {error_path}")
    print(metrics)


if __name__ == "__main__":
    main()
