from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from common import clip_xyxy, load_json, load_yaml, save_json, xywh_to_xyxy
from prepare_yolo_seg import decode_rle, mask_to_segments


def choose_crop_box(
    bbox: list[float],
    width: int,
    height: int,
    crop_size: int,
    context: float,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(float(crop_size), bw * context, bh * context)
    side = min(side, float(max(width, height)))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = int(round(left + side))
    bottom = int(round(top + side))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    return [left, top, right, bottom]


def ann_tags(ann: dict[str, Any], image_width: int, image_height: int, tiny_width: float, tiny_area: float, large_area: float) -> tuple[bool, bool, list[float]]:
    box = clip_xyxy(xywh_to_xyxy(ann["bbox"]), image_width, image_height)
    w = box[2] - box[0]
    h = box[3] - box[1]
    area = w * h
    return (w <= tiny_width or area <= tiny_area), area >= large_area, box


def crop_masks_to_label_lines(
    anns: list[dict[str, Any]],
    crop_box: list[int],
    epsilon_ratio: float,
    max_points: int,
    min_mask_pixels: int,
) -> list[str]:
    left, top, right, bottom = crop_box
    lines: list[str] = []
    for ann in anns:
        if "segmentation" not in ann:
            continue
        mask = decode_rle(ann["segmentation"])
        crop = mask[top:bottom, left:right]
        if int(crop.sum()) < min_mask_pixels:
            continue
        for segment in mask_to_segments(crop.astype(np.uint8), epsilon_ratio=epsilon_ratio, max_points=max_points):
            flat = segment.reshape(-1)
            if flat.size < 6:
                continue
            coords = " ".join(f"{value:.6f}" for value in flat.tolist())
            lines.append(f"0 {coords}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tiny/large crack crop training data for YOLO-seg.")
    parser.add_argument("--config", default="configs/yolo_seg_crack_hybrid.yaml")
    parser.add_argument("--out-suffix", default="scalecrop")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--context", type=float, default=2.5)
    parser.add_argument("--tiny-repeat", type=int, default=2)
    parser.add_argument("--large-repeat", type=int, default=1)
    parser.add_argument("--max-crops-per-image", type=int, default=4)
    parser.add_argument("--min-mask-pixels", type=int, default=3)
    parser.add_argument("--tiny-width", type=float, default=None)
    parser.add_argument("--tiny-area", type=float, default=None)
    parser.add_argument("--large-area", type=float, default=None)
    parser.add_argument("--epsilon-ratio", type=float, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_root = Path(cfg["dataset_root"])
    prepared_root = Path(cfg["prepared_root"])
    eval_cfg = cfg["eval"]
    seg_cfg = cfg.get("segmentation", {})
    tiny_width = args.tiny_width if args.tiny_width is not None else float(eval_cfg["tiny_width"])
    tiny_area = args.tiny_area if args.tiny_area is not None else float(eval_cfg["tiny_area"])
    large_area = args.large_area if args.large_area is not None else float(eval_cfg["large_area"])
    epsilon_ratio = args.epsilon_ratio if args.epsilon_ratio is not None else float(seg_cfg.get("epsilon_ratio", 0.0005))
    max_points = args.max_points if args.max_points is not None else int(seg_cfg.get("max_points", 400))

    manifest_path = prepared_root / "split_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found. Run src/prepare_yolo_seg.py first.")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    train_ids = set(manifest["train_ids"])
    items = [item for item in load_json(dataset_root / "trainval" / "trainval.json")["Dataset"] if item["ID"] in train_ids]

    crop_image_dir = prepared_root / "images" / f"train_{args.out_suffix}"
    crop_label_dir = prepared_root / "labels" / f"train_{args.out_suffix}"
    crop_image_dir.mkdir(parents=True, exist_ok=True)
    crop_label_dir.mkdir(parents=True, exist_ok=True)

    crop_paths: list[str] = []
    crop_records: list[dict[str, Any]] = []
    image_root = dataset_root / "trainval"
    for item in items:
        image_path = image_root / item["Image"]
        with Image.open(image_path) as im:
            image = im.convert("L")
        width, height = image.size
        crop_specs: list[tuple[str, list[int], int]] = []
        for ann_idx, ann in enumerate(item.get("Annotations", [])):
            is_tiny, is_large, box = ann_tags(ann, width, height, tiny_width, tiny_area, large_area)
            repeat = args.tiny_repeat if is_tiny else args.large_repeat if is_large else 0
            tag = "tiny" if is_tiny else "large" if is_large else ""
            for rep_idx in range(repeat):
                if len(crop_specs) >= args.max_crops_per_image:
                    break
                crop_specs.append((f"{tag}_a{ann_idx}_r{rep_idx}", choose_crop_box(box, width, height, args.crop_size, args.context), ann_idx))
        for spec_idx, (tag, crop_box, ann_idx) in enumerate(crop_specs):
            left, top, right, bottom = crop_box
            crop = image.crop((left, top, right, bottom))
            stem = Path(item["Image"]).stem
            crop_name = f"{stem}_{tag}_{spec_idx:02d}.png"
            crop_img_path = crop_image_dir / crop_name
            crop_label_path = crop_label_dir / f"{Path(crop_name).stem}.txt"
            lines = crop_masks_to_label_lines(
                item.get("Annotations", []),
                crop_box=crop_box,
                epsilon_ratio=epsilon_ratio,
                max_points=max_points,
                min_mask_pixels=args.min_mask_pixels,
            )
            if not lines:
                continue
            crop.save(crop_img_path)
            crop_label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            crop_paths.append(str(crop_img_path.resolve()))
            crop_records.append(
                {
                    "image_id": item["ID"],
                    "image": item["Image"],
                    "ann_idx": ann_idx,
                    "tag": tag,
                    "crop_box": crop_box,
                    "crop_image": str(crop_img_path),
                    "crop_label": str(crop_label_path),
                    "label_count": len(lines),
                }
            )

    base_train_txt = prepared_root / "train.txt"
    if not base_train_txt.exists():
        raise FileNotFoundError(f"{base_train_txt} not found. Run src/prepare_yolo_seg.py first.")
    base_paths = [line.strip() for line in base_train_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    crop_txt = prepared_root / f"train_{args.out_suffix}_only.txt"
    combined_txt = prepared_root / f"train_{args.out_suffix}.txt"
    crop_txt.write_text("\n".join(crop_paths) + ("\n" if crop_paths else ""), encoding="utf-8")
    combined_txt.write_text("\n".join(base_paths + crop_paths) + "\n", encoding="utf-8")

    base_data_yaml = prepared_root / "crack_seg.yaml"
    data_cfg = load_yaml(base_data_yaml)
    data_cfg["train"] = str(combined_txt.resolve())
    data_yaml = prepared_root / f"crack_seg_{args.out_suffix}.yaml"
    with data_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, allow_unicode=True, sort_keys=False)

    summary = {
        "config": args.config,
        "base_train_images": len(base_paths),
        "crop_images": len(crop_paths),
        "combined_train_rows": len(base_paths) + len(crop_paths),
        "tiny_crop_records": sum(1 for row in crop_records if row["tag"].startswith("tiny")),
        "large_crop_records": sum(1 for row in crop_records if row["tag"].startswith("large")),
        "crop_size": args.crop_size,
        "context": args.context,
        "tiny_repeat": args.tiny_repeat,
        "large_repeat": args.large_repeat,
        "max_crops_per_image": args.max_crops_per_image,
        "out_crop_txt": str(crop_txt),
        "out_train_txt": str(combined_txt),
        "out_data_yaml": str(data_yaml),
    }
    reports_dir = Path(cfg["outputs"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, reports_dir / f"scale_crop_{args.out_suffix}_summary.json")
    save_json(crop_records, reports_dir / f"scale_crop_{args.out_suffix}_items.json")
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
