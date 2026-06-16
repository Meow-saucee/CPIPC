from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import load_json, save_json


def box_area(pred: dict[str, Any]) -> float:
    return max(0.0, float(pred["x2"]) - float(pred["x1"])) * max(0.0, float(pred["y2"]) - float(pred["y1"]))


def box_aspect(pred: dict[str, Any]) -> float:
    width = max(1e-6, float(pred["x2"]) - float(pred["x1"]))
    height = max(1e-6, float(pred["y2"]) - float(pred["y1"]))
    return max(width, height) / max(1e-6, min(width, height))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore official-format submission boxes by geometry priors.")
    parser.add_argument("--submit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--large-area", type=float, default=90000.0)
    parser.add_argument("--long-aspect", type=float, default=3.0)
    parser.add_argument("--large-floor", type=float, default=0.0)
    parser.add_argument("--long-floor", type=float, default=0.0)
    parser.add_argument("--large-mult", type=float, default=1.0)
    parser.add_argument("--long-mult", type=float, default=1.0)
    parser.add_argument("--small-mult", type=float, default=1.0)
    parser.add_argument("--max-score", type=float, default=1.0)
    args = parser.parse_args()

    rows = load_json(args.submit)
    for row in rows:
        preds = row.get("predict_bboxes", [])
        for pred in preds:
            score = float(pred.get("score", 0.0))
            area = box_area(pred)
            aspect = box_aspect(pred)
            is_large = area >= args.large_area
            is_long = aspect >= args.long_aspect
            if is_large:
                score = max(score * args.large_mult, args.large_floor)
            if is_long:
                score = max(score * args.long_mult, args.long_floor)
            if not is_large and not is_long:
                score *= args.small_mult
            pred["score"] = min(args.max_score, max(0.0, score))
        preds.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        row["predict_bboxes"] = preds

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(rows, out_path)
    print(f"Saved rescored submission to {out_path}")
    print(f"rows={len(rows)}, preds={sum(len(row.get('predict_bboxes', [])) for row in rows)}")


if __name__ == "__main__":
    main()
