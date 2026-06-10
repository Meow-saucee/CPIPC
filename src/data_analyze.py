from __future__ import annotations

import argparse
import collections
import statistics
from pathlib import Path
from typing import Any

from PIL import Image

from common import clip_xywh, iter_images, load_json, quantiles, save_json


def summarize_images(split_dir: Path, items: list[dict[str, Any]]) -> dict[str, Any]:
    image_dir = split_dir / "images"
    listed = {item["Image"] for item in items}
    files = {str(p.relative_to(split_dir)) for p in iter_images(image_dir)}
    rows: list[dict[str, Any]] = []
    modes: collections.Counter[str] = collections.Counter()
    formats: collections.Counter[str] = collections.Counter()
    suffixes: collections.Counter[str] = collections.Counter()
    read_errors: list[str] = []

    for item in items:
        img_path = split_dir / item["Image"]
        try:
            with Image.open(img_path) as im:
                rows.append(
                    {
                        "id": item["ID"],
                        "image": item["Image"],
                        "width": im.width,
                        "height": im.height,
                        "area": im.width * im.height,
                        "mode": im.mode,
                        "format": im.format,
                    }
                )
                modes[im.mode] += 1
                formats[str(im.format)] += 1
                suffixes[img_path.suffix.lower()] += 1
        except Exception as exc:  # noqa: BLE001
            read_errors.append(f"{img_path}: {exc}")

    widths = [r["width"] for r in rows]
    heights = [r["height"] for r in rows]
    areas = [r["area"] for r in rows]
    return {
        "json_images": len(items),
        "image_files": len(files),
        "missing_images": sorted(list(listed - files)),
        "orphan_images": sorted(list(files - listed)),
        "read_errors": read_errors,
        "modes": dict(modes),
        "formats": dict(formats),
        "suffixes": dict(suffixes),
        "width_quantiles": quantiles(widths, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]),
        "height_quantiles": quantiles(heights, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]),
        "area_quantiles": quantiles(areas, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]),
        "scale_buckets": {
            "<=128": sum(max(r["width"], r["height"]) <= 128 for r in rows),
            "<=512": sum(max(r["width"], r["height"]) <= 512 for r in rows),
            "<=1024": sum(max(r["width"], r["height"]) <= 1024 for r in rows),
            "<=2048": sum(max(r["width"], r["height"]) <= 2048 for r in rows),
            ">2048": sum(max(r["width"], r["height"]) > 2048 for r in rows),
        },
        "smallest_images": sorted(rows, key=lambda r: r["area"])[:10],
        "largest_images": sorted(rows, key=lambda r: r["area"])[-10:],
    }


def summarize_annotations(dataset_root: Path, train_items: list[dict[str, Any]]) -> dict[str, Any]:
    ann_rows: list[dict[str, Any]] = []
    categories: collections.Counter[str] = collections.Counter()
    types: collections.Counter[str] = collections.Counter()
    per_image: collections.Counter[int] = collections.Counter()
    invalid: list[dict[str, Any]] = []
    rle_size_mismatch = 0

    for item in train_items:
        img_path = dataset_root / "trainval" / item["Image"]
        with Image.open(img_path) as im:
            width, height = im.width, im.height
        anns = item.get("Annotations", [])
        per_image[len(anns)] += 1
        for idx, ann in enumerate(anns):
            categories[str(ann.get("ObjectCategory"))] += 1
            types[str(ann.get("Type"))] += 1
            x, y, w, h = map(float, ann["bbox"])
            cx, cy, cw, ch = clip_xywh(x, y, w, h, width, height)
            if abs(cx - x) > 1e-6 or abs(cy - y) > 1e-6 or abs(cw - w) > 1e-6 or abs(ch - h) > 1e-6:
                invalid.append(
                    {
                        "id": item["ID"],
                        "image": item["Image"],
                        "ann_index": idx,
                        "image_size": [width, height],
                        "bbox": [x, y, w, h],
                        "clipped_bbox": [cx, cy, cw, ch],
                    }
                )
            seg_size = ann.get("segmentation", {}).get("size", [])
            if seg_size and list(seg_size) != [height, width]:
                rle_size_mismatch += 1
            ann_rows.append(
                {
                    "id": item["ID"],
                    "image": item["Image"],
                    "width": cw,
                    "height": ch,
                    "area": cw * ch,
                }
            )

    widths = [r["width"] for r in ann_rows]
    heights = [r["height"] for r in ann_rows]
    areas = [r["area"] for r in ann_rows]
    return {
        "annotation_count": len(ann_rows),
        "categories": dict(categories),
        "types": dict(types),
        "annotations_per_image": dict(sorted(per_image.items())),
        "rle_size_mismatch": rle_size_mismatch,
        "bbox_needs_clip_count": len(invalid),
        "bbox_needs_clip_examples": invalid[:20],
        "bbox_width_quantiles": quantiles(widths, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]),
        "bbox_height_quantiles": quantiles(heights, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]),
        "bbox_area_quantiles": quantiles(areas, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]),
        "bbox_width_mean": statistics.mean(widths) if widths else 0,
        "bbox_height_mean": statistics.mean(heights) if heights else 0,
        "bbox_area_mean": statistics.mean(areas) if areas else 0,
        "tiny_count_width_le_5_or_area_le_50": sum(r["width"] <= 5 or r["area"] <= 50 for r in ann_rows),
        "tiny_count_width_le_5": sum(r["width"] <= 5 for r in ann_rows),
        "tiny_count_area_le_50": sum(r["area"] <= 50 for r in ann_rows),
        "large_count_area_ge_90000": sum(r["area"] >= 90000 for r in ann_rows),
        "area_buckets": {
            "<=50": sum(a <= 50 for a in areas),
            "50-100": sum(50 < a <= 100 for a in areas),
            "100-1k": sum(100 < a <= 1000 for a in areas),
            "1k-10k": sum(1000 < a <= 10000 for a in areas),
            "10k-90k": sum(10000 < a < 90000 for a in areas),
            ">=90k": sum(a >= 90000 for a in areas),
        },
        "smallest_boxes": sorted(ann_rows, key=lambda r: r["area"])[:10],
        "largest_boxes": sorted(ann_rows, key=lambda r: r["area"])[-10:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CPIPC crack dataset.")
    parser.add_argument("--dataset", default="dataset", help="Dataset root containing trainval and test.")
    parser.add_argument("--out", default="outputs/reports/data_stats.json", help="Output JSON path.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    train_items = load_json(dataset_root / "trainval" / "trainval.json")["Dataset"]
    test_items = load_json(dataset_root / "test" / "test.json")["Dataset"]
    stats = {
        "dataset_root": str(dataset_root),
        "trainval": summarize_images(dataset_root / "trainval", train_items),
        "test": summarize_images(dataset_root / "test", test_items),
        "annotations": summarize_annotations(dataset_root, train_items),
    }
    save_json(stats, args.out)
    print(f"Saved dataset statistics to {args.out}")
    print(
        "Summary:",
        {
            "train_images": stats["trainval"]["json_images"],
            "test_images": stats["test"]["json_images"],
            "annotations": stats["annotations"]["annotation_count"],
            "tiny": stats["annotations"]["tiny_count_width_le_5_or_area_le_50"],
            "large": stats["annotations"]["large_count_area_ge_90000"],
        },
    )


if __name__ == "__main__":
    main()
