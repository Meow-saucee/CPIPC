from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common import clip_xyxy, load_json, load_yaml, nms_xyxy, round_box, save_json


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def predict_array(model: Any, image: np.ndarray, imgsz: int, conf: float, iou: float, max_det: int) -> list[dict[str, Any]]:
    results = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False)
    result = results[0]
    preds: list[dict[str, Any]] = []
    if result.boxes is None:
        return preds
    for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
        preds.append({"box": [float(v) for v in box], "score": float(score)})
    return preds


def resize_for_global(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(width, height))
    if scale >= 1.0:
        return image, 1.0
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
    return resized, scale


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def predict_multiscale(model: Any, image: np.ndarray, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    height, width = image.shape[:2]
    infer_cfg = cfg["infer"]
    boxes: list[list[float]] = []
    scores: list[float] = []

    start = time.perf_counter()
    if max(width, height) <= infer_cfg["direct_max_side"]:
        preds = predict_array(model, image, infer_cfg["imgsz"], infer_cfg["conf"], infer_cfg["iou"], infer_cfg["max_det"])
        for pred in preds:
            boxes.append(clip_xyxy(pred["box"], width, height))
            scores.append(pred["score"])
    else:
        if infer_cfg.get("include_global_for_large", True):
            global_img, scale = resize_for_global(image, infer_cfg["imgsz"])
            preds = predict_array(
                model,
                global_img,
                infer_cfg["imgsz"],
                infer_cfg["conf"],
                infer_cfg["iou"],
                infer_cfg["max_det"],
            )
            for pred in preds:
                box = [v / scale for v in pred["box"]]
                boxes.append(clip_xyxy(box, width, height))
                scores.append(pred["score"])

        tile = infer_cfg["tile_size"]
        overlap = infer_cfg["tile_overlap"]
        for y0 in tile_starts(height, tile, overlap):
            for x0 in tile_starts(width, tile, overlap):
                crop = image[y0 : y0 + tile, x0 : x0 + tile]
                preds = predict_array(
                    model,
                    crop,
                    infer_cfg["imgsz"],
                    infer_cfg["conf"],
                    infer_cfg["iou"],
                    infer_cfg["max_det"],
                )
                for pred in preds:
                    x1, y1, x2, y2 = pred["box"]
                    box = [x1 + x0, y1 + y0, x2 + x0, y2 + y0]
                    boxes.append(clip_xyxy(box, width, height))
                    scores.append(pred["score"])

    keep = nms_xyxy(boxes, scores, infer_cfg["iou"]) if boxes else []
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    predictions = [{"box": boxes[i], "score": scores[i]} for i in keep]
    predictions.sort(key=lambda x: x["score"], reverse=True)
    return predictions, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference and generate official results.json.")
    parser.add_argument("--config", default="configs/yolo_crack.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_root = Path(cfg["dataset_root"])
    submissions_dir = Path(cfg["outputs"]["submissions"])
    submissions_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing ultralytics. Install dependencies from requirements.txt first.") from exc

    model = YOLO(args.weights)
    json_path = dataset_root / args.split / f"{args.split}.json"
    items = load_json(json_path)["Dataset"]

    results = []
    for idx, item in enumerate(items, start=1):
        img_path = dataset_root / args.split / item["Image"]
        image = load_rgb(img_path)
        height, width = image.shape[:2]
        preds, elapsed_ms = predict_multiscale(model, image, cfg)
        predict_bboxes = []
        for pred in preds:
            x1, y1, x2, y2 = round_box(clip_xyxy(pred["box"], width, height))
            if x2 <= x1 or y2 <= y1:
                continue
            predict_bboxes.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": float(pred["score"]),
                    "label": "crack",
                }
            )
        results.append(
            {
                "ID": item["ID"],
                "image path": item["Image"],
                "inference_time_ms": round(elapsed_ms, 3),
                "groundtruth_bboxes": [],
                "predict_bboxes": predict_bboxes,
            }
        )
        if idx % 20 == 0 or idx == len(items):
            print(f"[{idx}/{len(items)}] processed")

    out_path = Path(args.out) if args.out else submissions_dir / "results.json"
    save_json(results, out_path)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
