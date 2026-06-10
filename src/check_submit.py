from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import image_size, load_json


REQUIRED_TOP_KEYS = {"ID", "image path", "inference_time_ms", "groundtruth_bboxes", "predict_bboxes"}
REQUIRED_BOX_KEYS = {"x1", "y1", "x2", "y2", "score", "label"}


def fail(errors: list[str], message: str) -> None:
    if len(errors) < 50:
        errors.append(message)


def validate_box(box: dict[str, Any], width: int, height: int, errors: list[str], prefix: str) -> None:
    missing = REQUIRED_BOX_KEYS - set(box)
    if missing:
        fail(errors, f"{prefix}: missing box keys {sorted(missing)}")
        return
    try:
        x1, y1, x2, y2 = [float(box[k]) for k in ["x1", "y1", "x2", "y2"]]
        score = float(box["score"])
    except Exception as exc:  # noqa: BLE001
        fail(errors, f"{prefix}: non-numeric bbox or score: {exc}")
        return
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        fail(errors, f"{prefix}: bbox outside image or invalid {[x1, y1, x2, y2]} for size {(width, height)}")
    if not 0 <= score <= 1:
        fail(errors, f"{prefix}: score out of range {score}")
    if box["label"] != "crack":
        fail(errors, f"{prefix}: label must be crack, got {box['label']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate official submission results.json.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--submit", default="outputs/submissions/results.json")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    submit_path = Path(args.submit)
    test_items = load_json(dataset_root / "test" / "test.json")["Dataset"]
    expected = {item["ID"]: item for item in test_items}
    results = load_json(submit_path)
    errors: list[str] = []

    if not isinstance(results, list):
        raise SystemExit("Submission must be a JSON list.")
    if len(results) != len(test_items):
        fail(errors, f"submission length {len(results)} != test length {len(test_items)}")

    seen_ids: set[int] = set()
    for row_idx, row in enumerate(results):
        if not isinstance(row, dict):
            fail(errors, f"row {row_idx}: must be object")
            continue
        missing = REQUIRED_TOP_KEYS - set(row)
        if missing:
            fail(errors, f"row {row_idx}: missing top keys {sorted(missing)}")
            continue
        image_id = row["ID"]
        if image_id not in expected:
            fail(errors, f"row {row_idx}: unexpected ID {image_id}")
            continue
        if image_id in seen_ids:
            fail(errors, f"row {row_idx}: duplicate ID {image_id}")
        seen_ids.add(image_id)
        item = expected[image_id]
        if row["image path"] != item["Image"]:
            fail(errors, f"ID {image_id}: image path {row['image path']} != {item['Image']}")
        if row["groundtruth_bboxes"] != []:
            fail(errors, f"ID {image_id}: groundtruth_bboxes must be []")
        try:
            infer_ms = float(row["inference_time_ms"])
            if infer_ms < 0:
                fail(errors, f"ID {image_id}: negative inference_time_ms")
        except Exception:  # noqa: BLE001
            fail(errors, f"ID {image_id}: inference_time_ms must be numeric")
        if not isinstance(row["predict_bboxes"], list):
            fail(errors, f"ID {image_id}: predict_bboxes must be list")
            continue
        width, height = image_size(dataset_root / "test" / item["Image"])
        for box_idx, box in enumerate(row["predict_bboxes"]):
            if not isinstance(box, dict):
                fail(errors, f"ID {image_id} box {box_idx}: must be object")
                continue
            validate_box(box, width, height, errors, f"ID {image_id} box {box_idx}")

    missing_ids = set(expected) - seen_ids
    if missing_ids:
        fail(errors, f"missing IDs: {sorted(list(missing_ids))[:20]}")

    if errors:
        print("Submission check failed:")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)
    print(f"Submission is valid: {submit_path}, rows={len(results)}")


if __name__ == "__main__":
    main()
