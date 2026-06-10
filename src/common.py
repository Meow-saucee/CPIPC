from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.width, im.height


def iter_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def clip_xywh(x: float, y: float, w: float, h: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(width), float(x)))
    y1 = max(0.0, min(float(height), float(y)))
    x2 = max(0.0, min(float(width), float(x) + float(w)))
    y2 = max(0.0, min(float(height), float(y) + float(h)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_xyxy(box: Iterable[float]) -> list[float]:
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def clip_xyxy(box: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_xyxy(boxes: list[list[float]], scores: list[float], iou_thr: float) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if box_iou(boxes[i], boxes[j]) <= iou_thr]
    return keep


def safe_symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def quantiles(values: list[float], points: Iterable[float]) -> dict[str, float]:
    if not values:
        return {}
    vals = sorted(values)
    out: dict[str, float] = {}
    for p in points:
        idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * p)))
        out[str(p)] = vals[idx]
    return out


def round_box(box: Iterable[float]) -> list[int]:
    x1, y1, x2, y2 = map(float, box)
    return [int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2))]
