from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from common import clip_xyxy, load_json, load_yaml, save_json, xywh_to_xyxy


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/prepare_yolo_seg.py first.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ann_scale_flags(
    item: dict[str, Any],
    image_root: Path,
    tiny_width: float,
    tiny_area: float,
    large_area: float,
    huge_side: int,
) -> dict[str, Any]:
    image_path = image_root / item["Image"]
    with Image.open(image_path) as im:
        width, height = im.width, im.height

    tiny_count = 0
    large_count = 0
    max_area = 0.0
    min_width = None
    for ann in item.get("Annotations", []):
        box = clip_xyxy(xywh_to_xyxy(ann["bbox"]), width, height)
        w = box[2] - box[0]
        h = box[3] - box[1]
        area = w * h
        max_area = max(max_area, area)
        min_width = w if min_width is None else min(min_width, w)
        if w <= tiny_width or area <= tiny_area:
            tiny_count += 1
        if area >= large_area:
            large_count += 1

    return {
        "image_id": item["ID"],
        "image": item["Image"],
        "width": width,
        "height": height,
        "max_side": max(width, height),
        "ann_count": len(item.get("Annotations", [])),
        "tiny_count": tiny_count,
        "large_count": large_count,
        "has_tiny": tiny_count > 0,
        "has_large": large_count > 0,
        "is_huge_image": max(width, height) >= huge_side,
        "max_ann_area": max_area,
        "min_ann_width": min_width,
    }


def repeat_factor(stats: dict[str, Any], args: argparse.Namespace) -> int:
    repeat = 1
    if stats["has_tiny"]:
        repeat = max(repeat, args.tiny_repeat)
    if stats["has_large"]:
        repeat = max(repeat, args.large_repeat)
    if stats["is_huge_image"]:
        repeat = max(repeat, args.huge_repeat)
    if stats["has_tiny"] and stats["has_large"]:
        repeat = max(repeat, args.tiny_large_repeat)
    return min(repeat, args.max_repeat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a scale-aware repeated train list for YOLO-seg training.")
    parser.add_argument("--config", default="configs/yolo_seg_crack_hybrid.yaml")
    parser.add_argument("--out-suffix", default="scaleaware")
    parser.add_argument("--tiny-width", type=float, default=None)
    parser.add_argument("--tiny-area", type=float, default=None)
    parser.add_argument("--large-area", type=float, default=None)
    parser.add_argument("--huge-side", type=int, default=2048)
    parser.add_argument("--tiny-repeat", type=int, default=3)
    parser.add_argument("--large-repeat", type=int, default=3)
    parser.add_argument("--huge-repeat", type=int, default=2)
    parser.add_argument("--tiny-large-repeat", type=int, default=4)
    parser.add_argument("--max-repeat", type=int, default=4)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_root = Path(cfg["dataset_root"])
    prepared_root = Path(cfg["prepared_root"])
    eval_cfg = cfg["eval"]
    tiny_width = args.tiny_width if args.tiny_width is not None else float(eval_cfg["tiny_width"])
    tiny_area = args.tiny_area if args.tiny_area is not None else float(eval_cfg["tiny_area"])
    large_area = args.large_area if args.large_area is not None else float(eval_cfg["large_area"])

    manifest = load_manifest(prepared_root / "split_manifest.yaml")
    train_ids = set(manifest["train_ids"])
    all_items = load_json(dataset_root / "trainval" / "trainval.json")["Dataset"]
    train_items = [item for item in all_items if item["ID"] in train_ids]
    image_root = dataset_root / "trainval"
    train_txt = prepared_root / "train.txt"
    if not train_txt.exists():
        raise FileNotFoundError(f"{train_txt} not found. Run src/prepare_yolo_seg.py first.")
    image_paths = [line.strip() for line in train_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    path_by_stem = {Path(path).stem: path for path in image_paths}

    repeated_paths: list[str] = []
    stats_rows: list[dict[str, Any]] = []
    repeats = Counter()
    for item in train_items:
        stats = ann_scale_flags(
            item,
            image_root=image_root,
            tiny_width=tiny_width,
            tiny_area=tiny_area,
            large_area=large_area,
            huge_side=args.huge_side,
        )
        rep = repeat_factor(stats, args)
        stem = Path(item["Image"]).stem
        if stem not in path_by_stem:
            raise KeyError(f"Image {item['Image']} not found in {train_txt}")
        repeated_paths.extend([path_by_stem[stem]] * rep)
        repeats[rep] += 1
        stats_rows.append({**stats, "repeat": rep})

    out_train = prepared_root / f"train_{args.out_suffix}.txt"
    out_train.write_text("\n".join(repeated_paths) + "\n", encoding="utf-8")

    base_data_yaml = prepared_root / "crack_seg.yaml"
    data_cfg = load_yaml(base_data_yaml)
    data_cfg["train"] = str(out_train.resolve())
    out_data_yaml = prepared_root / f"crack_seg_{args.out_suffix}.yaml"
    with out_data_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, allow_unicode=True, sort_keys=False)

    summary = {
        "config": args.config,
        "base_train_images": len(train_items),
        "repeated_train_rows": len(repeated_paths),
        "repeat_histogram": dict(sorted(repeats.items())),
        "tiny_images": sum(1 for row in stats_rows if row["has_tiny"]),
        "large_images": sum(1 for row in stats_rows if row["has_large"]),
        "huge_images": sum(1 for row in stats_rows if row["is_huge_image"]),
        "tiny_and_large_images": sum(1 for row in stats_rows if row["has_tiny"] and row["has_large"]),
        "tiny_width": tiny_width,
        "tiny_area": tiny_area,
        "large_area": large_area,
        "huge_side": args.huge_side,
        "out_train": str(out_train),
        "out_data_yaml": str(out_data_yaml),
    }
    reports_dir = Path(cfg["outputs"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, reports_dir / f"scale_aware_train_{args.out_suffix}_summary.json")
    save_json(stats_rows, reports_dir / f"scale_aware_train_{args.out_suffix}_items.json")

    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
