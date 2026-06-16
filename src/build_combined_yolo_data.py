from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from common import load_yaml, save_json


def read_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_path_for_image(prepared_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    parts = path.parts
    if "images" not in parts:
        raise ValueError(f"Image path does not contain an images directory: {image_path}")
    idx = parts.index("images")
    rel_after_images = Path(*parts[idx + 1 :])
    return prepared_root / "labels" / rel_after_images.with_suffix(".txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine YOLO train lists and write a matching data yaml.")
    parser.add_argument("--config", default="configs/yolo_seg_crack_hybrid.yaml")
    parser.add_argument("--base-data-yaml", default=None, help="Base data yaml to copy val/test/names/path from.")
    parser.add_argument("--out-suffix", required=True)
    parser.add_argument("--train-lists", nargs="+", required=True, help="Train txt files to concatenate in order.")
    parser.add_argument("--dedupe", action="store_true", help="Remove duplicate rows. Keep disabled for intentional oversampling.")
    parser.add_argument("--allow-missing-labels", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    prepared_root = Path(cfg["prepared_root"])
    base_data_yaml = Path(args.base_data_yaml) if args.base_data_yaml else prepared_root / "crack_seg.yaml"
    data_cfg: dict[str, Any] = load_yaml(base_data_yaml)

    rows: list[str] = []
    source_counts: dict[str, int] = {}
    for list_name in args.train_lists:
        path = Path(list_name)
        list_rows = read_list(path)
        rows.extend(list_rows)
        source_counts[str(path)] = len(list_rows)

    if args.dedupe:
        seen = set()
        deduped = []
        for row in rows:
            if row in seen:
                continue
            seen.add(row)
            deduped.append(row)
        rows = deduped

    missing_images = [row for row in rows if not Path(row).exists()]
    missing_labels = []
    empty_labels = []
    for row in rows:
        label_path = label_path_for_image(prepared_root, row)
        if not label_path.exists():
            missing_labels.append(str(label_path))
        elif label_path.stat().st_size == 0:
            empty_labels.append(str(label_path))
    if missing_images:
        raise FileNotFoundError(f"Missing images: {missing_images[:5]} ... total={len(missing_images)}")
    if missing_labels and not args.allow_missing_labels:
        raise FileNotFoundError(f"Missing labels: {missing_labels[:5]} ... total={len(missing_labels)}")

    out_train = prepared_root / f"train_{args.out_suffix}.txt"
    out_train.write_text("\n".join(rows) + "\n", encoding="utf-8")
    data_cfg["train"] = str(out_train.resolve())
    out_data_yaml = prepared_root / f"crack_seg_{args.out_suffix}.yaml"
    with out_data_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, allow_unicode=True, sort_keys=False)

    summary = {
        "config": args.config,
        "base_data_yaml": str(base_data_yaml),
        "train_lists": source_counts,
        "dedupe": args.dedupe,
        "combined_rows": len(rows),
        "unique_rows": len(set(rows)),
        "missing_images": len(missing_images),
        "missing_labels": len(missing_labels),
        "empty_labels": len(empty_labels),
        "out_train": str(out_train),
        "out_data_yaml": str(out_data_yaml),
    }
    reports_dir = Path(cfg["outputs"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, reports_dir / f"combined_train_{args.out_suffix}_summary.json")
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
