from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from pycocotools import mask as mask_utils

from common import load_json, load_yaml, safe_link_or_copy
from prepare_yolo import stratified_split


def decode_rle(segmentation: dict[str, Any]) -> np.ndarray:
    counts = segmentation["counts"]
    rle = {"size": segmentation["size"], "counts": counts.encode("utf-8")} if isinstance(counts, str) else segmentation
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint8)


def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int32)
    return points[indices]


def mask_to_segments(mask: np.ndarray, epsilon_ratio: float, max_points: int) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    height, width = mask.shape[:2]
    segments: list[np.ndarray] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0 else contour
        points = approx.reshape(-1, 2).astype(np.float32)
        points = downsample_points(points, max_points=max_points)
        points[:, 0] /= width
        points[:, 1] /= height
        points = np.clip(points, 0.0, 1.0)
        if points.shape[0] >= 3:
            segments.append(points)
    return segments


def label_lines(item: dict[str, Any], epsilon_ratio: float, max_points: int) -> list[str]:
    lines: list[str] = []
    for ann in item.get("Annotations", []):
        if "segmentation" not in ann:
            continue
        mask = decode_rle(ann["segmentation"])
        for segment in mask_to_segments(mask, epsilon_ratio=epsilon_ratio, max_points=max_points):
            flat = segment.reshape(-1)
            if flat.size < 6:
                continue
            coords = " ".join(f"{value:.6f}" for value in flat.tolist())
            lines.append(f"0 {coords}")
    return lines


def write_split(
    split_name: str,
    items: list[dict[str, Any]],
    source_split_dir: Path,
    out_root: Path,
    with_labels: bool,
    epsilon_ratio: float,
    max_points: int,
) -> Path:
    image_out_dir = out_root / "images" / split_name
    label_out_dir = out_root / "labels" / split_name
    image_out_dir.mkdir(parents=True, exist_ok=True)
    if with_labels:
        label_out_dir.mkdir(parents=True, exist_ok=True)

    list_path = out_root / f"{split_name}.txt"
    rows: list[str] = []
    for item in items:
        src_img = source_split_dir / item["Image"]
        dst_img = image_out_dir / Path(item["Image"]).name
        safe_link_or_copy(src_img, dst_img)
        rows.append(str(dst_img.resolve()))
        if with_labels:
            label_path = label_out_dir / f"{Path(item['Image']).stem}.txt"
            lines = label_lines(item, epsilon_ratio=epsilon_ratio, max_points=max_points)
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    list_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return list_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official RLE masks to YOLO segmentation format.")
    parser.add_argument("--config", default="configs/yolo_seg_crack.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epsilon-ratio", type=float, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_root = Path(args.dataset or cfg["dataset_root"])
    out_root = Path(args.out or cfg["prepared_root"])
    split_cfg = cfg.get("split", {})
    seg_cfg = cfg.get("segmentation", {})
    val_ratio = args.val_ratio if args.val_ratio is not None else split_cfg.get("val_ratio", 0.2)
    seed = args.seed if args.seed is not None else split_cfg.get("seed", 42)
    epsilon_ratio = args.epsilon_ratio if args.epsilon_ratio is not None else seg_cfg.get("epsilon_ratio", 0.0005)
    max_points = args.max_points if args.max_points is not None else seg_cfg.get("max_points", 400)

    train_items = load_json(dataset_root / "trainval" / "trainval.json")["Dataset"]
    test_items = load_json(dataset_root / "test" / "test.json")["Dataset"]
    train_items, val_items = stratified_split(train_items, val_ratio=val_ratio, seed=seed)

    train_list = write_split("train", train_items, dataset_root / "trainval", out_root, True, epsilon_ratio, max_points)
    val_list = write_split("val", val_items, dataset_root / "trainval", out_root, True, epsilon_ratio, max_points)
    test_list = write_split("test", test_items, dataset_root / "test", out_root, False, epsilon_ratio, max_points)

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": str(train_list.resolve()),
        "val": str(val_list.resolve()),
        "test": str(test_list.resolve()),
        "names": {0: "crack"},
    }
    with (out_root / "crack_seg.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    split_manifest = {
        "seed": seed,
        "val_ratio": val_ratio,
        "epsilon_ratio": epsilon_ratio,
        "max_points": max_points,
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
        "train_ids": [item["ID"] for item in train_items],
        "val_ids": [item["ID"] for item in val_items],
    }
    with (out_root / "split_manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(split_manifest, f, allow_unicode=True, sort_keys=False)

    print(f"Prepared YOLO segmentation data under {out_root}")
    print(f"train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    print(f"YOLO-seg data config: {out_root / 'crack_seg.yaml'}")


if __name__ == "__main__":
    main()
