from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import load_json, save_json


def parse_ids(text: str) -> set[int]:
    ids: set[int] = set()
    for part in text.replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        ids.add(int(part))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Route selected image IDs from an alternate submission.")
    parser.add_argument("--base", required=True, help="Default submission.")
    parser.add_argument("--alternate", required=True, help="Submission used for selected IDs.")
    parser.add_argument("--ids", required=True, help="Comma-separated IDs to replace from alternate.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ids = parse_ids(args.ids)
    base_rows = load_json(args.base)
    alt_rows = load_json(args.alternate)
    alt_by_id = {int(row["ID"]): row for row in alt_rows}
    out_rows: list[dict[str, Any]] = []
    replaced = 0
    for row in base_rows:
        image_id = int(row["ID"])
        if image_id in ids:
            if image_id not in alt_by_id:
                raise ValueError(f"alternate submission missing ID {image_id}")
            out_rows.append(dict(alt_by_id[image_id]))
            replaced += 1
        else:
            out_rows.append(dict(row))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_rows, out_path)
    print(f"Saved routed submission to {out_path}")
    print(f"rows={len(out_rows)}, replaced={replaced}, ids={sorted(ids)}")


if __name__ == "__main__":
    main()
