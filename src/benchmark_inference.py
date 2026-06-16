from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from common import load_json, load_yaml, save_json
from infer_submit_seg import load_rgb, load_split_items, predict_multiscale


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def select_items(items: list[dict[str, Any]], ids: set[int], image_names: set[str], limit: int | None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in items:
        image_id = int(item["ID"])
        image_name = str(item["Image"])
        if ids and image_id not in ids:
            continue
        if image_names and image_name not in image_names:
            continue
        selected.append(item)
    if not ids and not image_names:
        selected = items
    if limit is not None:
        selected = selected[:limit]
    return selected


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark real YOLO-seg inference time on selected images.")
    parser.add_argument("--config", default="configs/yolo_seg_crack_hybrid.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--ids", default="", help="Comma-separated image IDs to benchmark.")
    parser.add_argument("--images", default="", help="Comma-separated image paths, e.g. images/3.jpg.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out-json", default="outputs/reports/benchmark_inference.json")
    parser.add_argument("--out-csv", default="outputs/reports/benchmark_inference.csv")
    parser.add_argument("--device", default=None, help="Optional Ultralytics device override, e.g. 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--direct-resize-max-side", type=int, default=None)
    parser.add_argument("--global-max-side", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--prediction-box-source", choices=["mask_box", "det_box", "prefer_mask"], default=None)
    parser.add_argument("--retina-masks", action="store_true", default=None)
    parser.add_argument("--no-retina-masks", action="store_false", dest="retina_masks")
    parser.add_argument("--keep-masks-for-merge", action="store_true", default=None)
    parser.add_argument("--no-keep-masks-for-merge", action="store_false", dest="keep_masks_for_merge")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    infer_cfg = cfg["infer"]
    overrides = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "direct_resize_max_side": args.direct_resize_max_side,
        "global_max_side": args.global_max_side,
        "tile_size": args.tile_size,
        "prediction_box_source": args.prediction_box_source,
        "retina_masks": args.retina_masks,
        "keep_masks_for_merge": args.keep_masks_for_merge,
    }
    for key, value in overrides.items():
        if value is not None:
            infer_cfg[key] = value
    if args.device is not None:
        cfg.setdefault("train", {})["device"] = args.device
    dataset_root = Path(cfg["dataset_root"])
    prepared_root = Path(cfg["prepared_root"])

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing ultralytics. Install dependencies from requirements.txt first.") from exc

    model = YOLO(args.weights)
    predict_device = args.device
    torch_device = None
    if args.device is not None:
        torch_device = f"cuda:{args.device}" if args.device.isdigit() else args.device
        model.to(torch_device)

    items, image_root = load_split_items(dataset_root, prepared_root, args.split)
    ids = {int(value) for value in args.ids.split(",") if value.strip()}
    image_names = {value.strip() for value in args.images.split(",") if value.strip()}
    selected = select_items(items, ids, image_names, args.limit)
    if not selected:
        raise SystemExit("No images selected for benchmark.")

    first_image = load_rgb(image_root / selected[0]["Image"])
    for _ in range(max(0, args.warmup)):
        _preds, _elapsed = predict_multiscale(model, first_image, cfg)
    sync_cuda()

    rows: list[dict[str, Any]] = []
    all_times: list[float] = []
    for item in selected:
        image_path = image_root / item["Image"]
        image = load_rgb(image_path)
        height, width = image.shape[:2]
        times: list[float] = []
        pred_count = 0
        for _ in range(args.repeat):
            sync_cuda()
            start = time.perf_counter()
            preds, _internal_ms = predict_multiscale(model, image, cfg)
            sync_cuda()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            times.append(elapsed_ms)
            pred_count = len(preds)
        all_times.extend(times)
        row = {
            "ID": int(item["ID"]),
            "image": item["Image"],
            "width": width,
            "height": height,
            "max_side": max(width, height),
            "pred_count": pred_count,
            **summarize(times),
            "times_ms": times,
        }
        rows.append(row)
        print(
            f"ID={row['ID']} image={row['image']} "
            f"mean={row['mean_ms']:.3f}ms max={row['max_ms']:.3f}ms preds={pred_count}",
            flush=True,
        )

    report = {
        "config": args.config,
        "weights": args.weights,
        "split": args.split,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "device": args.device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "overall": summarize(all_times),
        "images": rows,
    }
    save_json(report, args.out_json)

    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["ID", "image", "width", "height", "max_side", "pred_count", "count", "mean_ms", "median_ms", "min_ms", "max_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"Saved benchmark JSON to {args.out_json}")
    print(f"Saved benchmark CSV to {args.out_csv}")
    print(f"Overall: {report['overall']}")


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent))
    main()
