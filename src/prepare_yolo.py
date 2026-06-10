from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from common import clip_xywh, load_json, safe_link_or_copy


def image_flags(item: dict[str, Any]) -> tuple[bool, bool]:
    tiny = False
    large = False
    for ann in item.get("Annotations", []):
        _, _, w, h = map(float, ann["bbox"])
        area = w * h
        tiny = tiny or w <= 5 or area <= 50
        large = large or area >= 90000
    return tiny, large


def stratified_split(items: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {"tiny": [], "large": [], "normal": []}
    for item in items:
        tiny, large = image_flags(item)
        if tiny:
            groups["tiny"].append(item)
        elif large:
            groups["large"].append(item)
        else:
            groups["normal"].append(item)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for group_items in groups.values():
        group_items = list(group_items)
        rng.shuffle(group_items)
        n_val = max(1, round(len(group_items) * val_ratio)) if group_items else 0
        val.extend(group_items[:n_val])
        train.extend(group_items[n_val:])
    train.sort(key=lambda x: x["ID"])
    val.sort(key=lambda x: x["ID"])
    return train, val


def yolo_label_lines(item: dict[str, Any], image_path: Path) -> list[str]:
    with Image.open(image_path) as im:
        width, height = im.width, im.height
    lines: list[str] = []
    for ann in item.get("Annotations", []):
        x, y, w, h = clip_xywh(*map(float, ann["bbox"]), width=width, height=height)
        if w <= 0 or h <= 0:
            continue
        xc = (x + w / 2.0) / width
        yc = (y + h / 2.0) / height
        nw = w / width
        nh = h / height
        vals = [0, xc, yc, nw, nh]
        lines.append(" ".join(f"{v:.8f}" if i else str(v) for i, v in enumerate(vals)))
    return lines


def write_split(
    split_name: str,
    items: list[dict[str, Any]],
    source_split_dir: Path,
    out_root: Path,
    with_labels: bool,
) -> Path:
    image_out_dir = out_root / "images" / split_name
    label_out_dir = out_root / "labels" / split_name
    image_out_dir.mkdir(parents=True, exist_ok=True)
    if with_labels:
        label_out_dir.mkdir(parents=True, exist_ok=True)

    list_path = out_root / f"{split_name}.txt"
    lines: list[str] = []
    for item in items:
        src_img = source_split_dir / item["Image"]
        dst_img = image_out_dir / Path(item["Image"]).name
        safe_link_or_copy(src_img, dst_img)
        lines.append(str(dst_img.resolve()))
        if with_labels:
            label_path = label_out_dir / f"{Path(item['Image']).stem}.txt"
            label_lines = yolo_label_lines(item, src_img)
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official crack JSON annotations to YOLO format.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--out", default="data/yolo")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_root = Path(args.out)
    train_items = load_json(dataset_root / "trainval" / "trainval.json")["Dataset"]
    test_items = load_json(dataset_root / "test" / "test.json")["Dataset"]
    train_items, val_items = stratified_split(train_items, args.val_ratio, args.seed)

    train_list = write_split("train", train_items, dataset_root / "trainval", out_root, with_labels=True)
    val_list = write_split("val", val_items, dataset_root / "trainval", out_root, with_labels=True)
    test_list = write_split("test", test_items, dataset_root / "test", out_root, with_labels=False)

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": str(train_list.resolve()),
        "val": str(val_list.resolve()),
        "test": str(test_list.resolve()),
        "names": {0: "crack"},
    }
    with (out_root / "crack.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    split_manifest = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
        "train_ids": [item["ID"] for item in train_items],
        "val_ids": [item["ID"] for item in val_items],
    }
    with (out_root / "split_manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(split_manifest, f, allow_unicode=True, sort_keys=False)

    print(f"Prepared YOLO data under {out_root}")
    print(f"train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    print(f"YOLO data config: {out_root / 'crack.yaml'}")


if __name__ == "__main__":
    main()
