from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


def check_label_file(path: Path) -> dict[str, Any]:
    result = {
        "path": str(path),
        "exists": path.exists(),
        "empty": False,
        "line_count": 0,
        "bad_lines": [],
        "coord_min": None,
        "coord_max": None,
    }
    if not path.exists():
        return result
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        result["empty"] = True
        return result
    coord_min = 1.0
    coord_max = 0.0
    for line_idx, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        result["line_count"] += 1
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            result["bad_lines"].append({"line": line_idx, "reason": "invalid_polygon_length", "text": line[:120]})
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            result["bad_lines"].append({"line": line_idx, "reason": "non_numeric", "text": line[:120]})
            continue
        if cls != 0:
            result["bad_lines"].append({"line": line_idx, "reason": f"unexpected_class_{cls}", "text": line[:120]})
        local_min = min(coords)
        local_max = max(coords)
        coord_min = min(coord_min, local_min)
        coord_max = max(coord_max, local_max)
        if local_min < -1e-6 or local_max > 1.0 + 1e-6:
            result["bad_lines"].append(
                {
                    "line": line_idx,
                    "reason": "coord_out_of_range",
                    "coord_min": local_min,
                    "coord_max": local_max,
                    "text": line[:120],
                }
            )
    result["coord_min"] = coord_min
    result["coord_max"] = coord_max
    return result


def check_split(name: str, list_path: Path, prepared_root: Path, require_labels: bool) -> dict[str, Any]:
    rows = read_list(list_path)
    missing_images: list[str] = []
    missing_labels: list[str] = []
    empty_labels: list[str] = []
    bad_label_files: list[dict[str, Any]] = []
    label_line_count = 0
    for row in rows:
        if not Path(row).exists():
            missing_images.append(row)
        if require_labels:
            label_path = label_path_for_image(prepared_root, row)
            label_result = check_label_file(label_path)
            if not label_result["exists"]:
                missing_labels.append(str(label_path))
            elif label_result["empty"]:
                empty_labels.append(str(label_path))
            elif label_result["bad_lines"]:
                bad_label_files.append(label_result)
            label_line_count += int(label_result["line_count"])
    return {
        "split": name,
        "list_path": str(list_path),
        "rows": len(rows),
        "unique_rows": len(set(rows)),
        "duplicate_rows": len(rows) - len(set(rows)),
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "bad_label_files": bad_label_files,
        "label_line_count": label_line_count,
    }


def resolve_list(data_cfg: dict[str, Any], key: str) -> Path:
    value = data_cfg.get(key)
    if value is None:
        raise KeyError(f"Missing '{key}' in data yaml")
    return Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO-seg data yaml before long training.")
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--allow-empty-labels", action="store_true")
    parser.add_argument("--allow-train-val-overlap", action="store_true")
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    data_cfg = load_yaml(data_yaml)
    prepared_root = Path(data_cfg.get("path", data_yaml.parent))
    train_list = resolve_list(data_cfg, "train")
    val_list = resolve_list(data_cfg, "val")
    test_list = resolve_list(data_cfg, "test") if data_cfg.get("test") else None

    train = check_split("train", train_list, prepared_root, require_labels=True)
    val = check_split("val", val_list, prepared_root, require_labels=True)
    test = check_split("test", test_list, prepared_root, require_labels=False) if test_list else None
    train_rows = set(read_list(train_list))
    val_rows = set(read_list(val_list))
    overlap = sorted(train_rows & val_rows)

    failures: list[str] = []
    for split in [train, val]:
        if split["missing_images"]:
            failures.append(f"{split['split']} missing_images={len(split['missing_images'])}")
        if split["missing_labels"]:
            failures.append(f"{split['split']} missing_labels={len(split['missing_labels'])}")
        if split["bad_label_files"]:
            failures.append(f"{split['split']} bad_label_files={len(split['bad_label_files'])}")
        if split["empty_labels"] and not args.allow_empty_labels:
            failures.append(f"{split['split']} empty_labels={len(split['empty_labels'])}")
    if test and test["missing_images"]:
        failures.append(f"test missing_images={len(test['missing_images'])}")
    if overlap and not args.allow_train_val_overlap:
        failures.append(f"train_val_overlap={len(overlap)}")

    summary: dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "prepared_root": str(prepared_root),
        "names": data_cfg.get("names"),
        "train": train,
        "val": val,
        "test": test,
        "train_val_overlap": overlap,
        "ok": not failures,
        "failures": failures,
    }
    out_path = Path(args.out) if args.out else Path("outputs/reports") / f"check_{data_yaml.stem}.json"
    save_json(summary, out_path)
    print(f"Saved data check to {out_path}")
    print(
        {
            "ok": summary["ok"],
            "train_rows": train["rows"],
            "train_unique": train["unique_rows"],
            "train_duplicates": train["duplicate_rows"],
            "val_rows": val["rows"],
            "test_rows": test["rows"] if test else None,
            "failures": failures,
        }
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
