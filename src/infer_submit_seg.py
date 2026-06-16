from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from PIL import Image

from common import clip_xyxy, load_json, load_yaml, round_box, save_json


def sync_cuda_if_available() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


@dataclass
class SegPrediction:
    box: list[float]
    score: float
    mask: np.ndarray | None = None
    mask_box: list[float] | None = None
    source: str = "model"


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def load_split_items(dataset_root: Path, prepared_root: Path, split: str) -> tuple[list[dict[str, Any]], Path]:
    if split == "test":
        return load_json(dataset_root / "test" / "test.json")["Dataset"], dataset_root / "test"
    if split == "val":
        items = load_json(dataset_root / "trainval" / "trainval.json")["Dataset"]
        manifest_path = prepared_root / "split_manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} not found. Run data preparation first.")
        with manifest_path.open("r", encoding="utf-8") as f:
            val_ids = set(yaml.safe_load(f)["val_ids"])
        return [item for item in items if item["ID"] in val_ids], dataset_root / "trainval"
    raise ValueError(f"Unsupported split: {split}")


def mask_to_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def expand_box(
    box: list[float],
    width: int,
    height: int,
    ratio: float = 0.0,
    pixels: float = 0.0,
    min_area: float = 0.0,
    min_side: float = 0.0,
) -> list[float]:
    if ratio <= 0 and pixels <= 0:
        return clip_xyxy(box, width, height)
    x1, y1, x2, y2 = map(float, box)
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if min_area > 0 and box_w * box_h < min_area:
        return clip_xyxy(box, width, height)
    if min_side > 0 and max(box_w, box_h) < min_side:
        return clip_xyxy(box, width, height)
    pad_x = pixels + box_w * ratio
    pad_y = pixels + box_h * ratio
    return clip_xyxy([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)


def expand_tiny_box(
    box: list[float],
    width: int,
    height: int,
    min_width: float = 0.0,
    min_height: float = 0.0,
    max_area: float = 0.0,
    min_side: float = 0.0,
    max_width: float = 0.0,
) -> list[float]:
    if min_width <= 0 and min_height <= 0:
        return clip_xyxy(box, width, height)
    x1, y1, x2, y2 = map(float, box)
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if max_area > 0 and box_w * box_h > max_area:
        return clip_xyxy(box, width, height)
    if min_side > 0 and max(box_w, box_h) < min_side:
        return clip_xyxy(box, width, height)
    if max_width > 0 and box_w > max_width:
        return clip_xyxy(box, width, height)
    target_w = max(box_w, min_width)
    target_h = max(box_h, min_height)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return clip_xyxy([cx - target_w / 2.0, cy - target_h / 2.0, cx + target_w / 2.0, cy + target_h / 2.0], width, height)


def expand_elongated_box(
    box: list[float],
    width: int,
    height: int,
    ratio: float = 0.0,
    pixels: float = 0.0,
    min_area: float = 0.0,
    min_aspect: float = 0.0,
) -> list[float]:
    if ratio <= 0 and pixels <= 0:
        return clip_xyxy(box, width, height)
    x1, y1, x2, y2 = map(float, box)
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if min_area > 0 and box_w * box_h < min_area:
        return clip_xyxy(box, width, height)
    short = max(1e-6, min(box_w, box_h))
    long = max(box_w, box_h)
    if min_aspect > 0 and long / short < min_aspect:
        return clip_xyxy(box, width, height)
    pad = pixels + long * ratio
    if box_w >= box_h:
        return clip_xyxy([x1 - pad, y1, x2 + pad, y2], width, height)
    return clip_xyxy([x1, y1 - pad, x2, y2 + pad], width, height)


def expand_edge_anchored_box(
    box: list[float],
    width: int,
    height: int,
    min_area: float = 0.0,
    min_aspect: float = 0.0,
    edge_margin: float = 0.0,
) -> list[float]:
    if edge_margin <= 0:
        return clip_xyxy(box, width, height)
    x1, y1, x2, y2 = map(float, box)
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if min_area > 0 and box_w * box_h < min_area:
        return clip_xyxy(box, width, height)
    short = max(1e-6, min(box_w, box_h))
    long = max(box_w, box_h)
    if min_aspect > 0 and long / short < min_aspect:
        return clip_xyxy(box, width, height)
    if box_w >= box_h:
        if x1 <= edge_margin:
            x1 = 0.0
        if width - x2 <= edge_margin:
            x2 = float(width)
    else:
        if y1 <= edge_margin:
            y1 = 0.0
        if height - y2 <= edge_margin:
            y2 = float(height)
    return clip_xyxy([x1, y1, x2, y2], width, height)


def scale_box_around_center(
    box: list[float],
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    new_w = box_w * scale_x
    new_h = box_h * scale_y
    return clip_xyxy([cx - new_w / 2.0, cy - new_h / 2.0, cx + new_w / 2.0, cy + new_h / 2.0], width, height)


def union_boxes(boxes: list[list[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def projection_gap(a1: float, a2: float, b1: float, b2: float) -> float:
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0.0


def projection_overlap_ratio(a1: float, a2: float, b1: float, b2: float) -> float:
    overlap = max(0.0, min(a2, b2) - max(a1, b1))
    denom = max(1e-6, min(a2 - a1, b2 - b1))
    return overlap / denom


def add_elongated_cluster_unions(preds: list[SegPrediction], width: int, height: int, cfg: dict[str, Any]) -> list[SegPrediction]:
    if not bool(cfg.get("union_elongated_clusters", False)):
        return preds

    box_source = str(cfg.get("union_cluster_box_source", "box")).lower()
    if box_source not in {"box", "mask_box", "prefer_mask"}:
        raise ValueError("union_cluster_box_source must be one of: box, mask_box, prefer_mask")
    min_area = float(cfg.get("union_cluster_min_area", 30000.0))
    min_aspect = float(cfg.get("union_cluster_min_aspect", 3.0))
    max_gap = float(cfg.get("union_cluster_max_gap", 256.0))
    min_cross_overlap = float(cfg.get("union_cluster_min_cross_overlap", 0.15))
    min_members = int(cfg.get("union_cluster_min_members", 2))
    score_factor = float(cfg.get("union_cluster_score_factor", 0.9))
    score_floor = float(cfg.get("union_cluster_score_floor", 0.0))
    score_area_norm = float(cfg.get("union_cluster_score_area_norm", 0.0))
    max_unions = int(cfg.get("union_cluster_max_new_boxes", 20))

    candidates: list[tuple[int, str, list[float], float]] = []
    for idx, pred in enumerate(preds):
        if box_source == "mask_box":
            source_box = pred.mask_box
        elif box_source == "prefer_mask":
            source_box = pred.mask_box or pred.box
        else:
            source_box = pred.box
        if source_box is None:
            continue
        x1, y1, x2, y2 = source_box
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        area = box_w * box_h
        short = max(1e-6, min(box_w, box_h))
        long = max(box_w, box_h)
        if area < min_area and long / short < min_aspect:
            continue
        orientation = "h" if box_w >= box_h else "v"
        candidates.append((idx, orientation, source_box, pred.score))

    parent = list(range(len(candidates)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def unite(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(candidates)):
        _, orient_i, box_i, _ = candidates[i]
        for j in range(i + 1, len(candidates)):
            _, orient_j, box_j, _ = candidates[j]
            if orient_i != orient_j:
                continue
            if orient_i == "h":
                cross_overlap = projection_overlap_ratio(box_i[1], box_i[3], box_j[1], box_j[3])
                main_gap = projection_gap(box_i[0], box_i[2], box_j[0], box_j[2])
            else:
                cross_overlap = projection_overlap_ratio(box_i[0], box_i[2], box_j[0], box_j[2])
                main_gap = projection_gap(box_i[1], box_i[3], box_j[1], box_j[3])
            if cross_overlap >= min_cross_overlap and main_gap <= max_gap:
                unite(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(candidates)):
        groups.setdefault(find(i), []).append(i)

    additions: list[SegPrediction] = []
    for group in groups.values():
        if len(group) < min_members:
            continue
        boxes = [candidates[i][2] for i in group]
        merged = clip_xyxy(union_boxes(boxes), width, height)
        if merged[2] <= merged[0] or merged[3] <= merged[1]:
            continue
        score = max(candidates[i][3] for i in group) * score_factor
        if score_floor > 0:
            if score_area_norm > 0:
                area = max(0.0, merged[2] - merged[0]) * max(0.0, merged[3] - merged[1])
                area_weight = min(1.0, area / score_area_norm)
                score = max(score, score_floor * area_weight)
            else:
                score = max(score, score_floor)
        additions.append(SegPrediction(box=merged, score=score, mask=None, mask_box=merged, source="union"))

    additions.sort(key=lambda item: item.score, reverse=True)
    if max_unions > 0:
        additions = additions[:max_unions]
    return preds + additions


def add_large_box_variants(preds: list[SegPrediction], width: int, height: int, cfg: dict[str, Any]) -> list[SegPrediction]:
    if not bool(cfg.get("large_box_variants", False)):
        return preds

    min_area = float(cfg.get("large_variant_min_area", 90000.0))
    score_factor = float(cfg.get("large_variant_score_factor", 0.35))
    score_floor = float(cfg.get("large_variant_score_floor", 0.0))
    max_new = int(cfg.get("large_variant_max_new_boxes", 50))
    expand_ratio = float(cfg.get("large_variant_expand_ratio", 0.0))
    shrink_ratio = float(cfg.get("large_variant_shrink_ratio", 0.0))
    directional_ratio = float(cfg.get("large_variant_directional_expand_ratio", 0.0))
    min_delta = float(cfg.get("large_variant_min_delta", 1.0))
    source_filter = str(cfg.get("large_variant_source_filter", "all")).lower()
    if source_filter not in {"all", "union", "model"}:
        raise ValueError("large_variant_source_filter must be one of: all, union, model")

    additions: list[SegPrediction] = []
    seen: set[tuple[int, int, int, int]] = set()

    def append_variant(source: SegPrediction, variant_box: list[float]) -> None:
        if variant_box[2] <= variant_box[0] or variant_box[3] <= variant_box[1]:
            return
        key = tuple(round(v) for v in variant_box)
        if key in seen:
            return
        if box_iou(source.box, variant_box) >= 0.999:
            return
        if (
            abs((variant_box[2] - variant_box[0]) - (source.box[2] - source.box[0])) < min_delta
            and abs((variant_box[3] - variant_box[1]) - (source.box[3] - source.box[1])) < min_delta
        ):
            return
        seen.add(key)
        score = max(source.score * score_factor, score_floor)
        additions.append(SegPrediction(box=variant_box, score=score, mask=None, mask_box=variant_box, source="large_variant"))

    for pred in sorted(preds, key=lambda item: item.score, reverse=True):
        if source_filter != "all" and pred.source != source_filter:
            continue
        x1, y1, x2, y2 = pred.box
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        if box_w * box_h < min_area:
            continue
        if expand_ratio > 0:
            append_variant(pred, scale_box_around_center(pred.box, width, height, 1.0 + expand_ratio, 1.0 + expand_ratio))
        if shrink_ratio > 0:
            append_variant(pred, scale_box_around_center(pred.box, width, height, max(0.01, 1.0 - shrink_ratio), max(0.01, 1.0 - shrink_ratio)))
        if directional_ratio > 0:
            append_variant(pred, scale_box_around_center(pred.box, width, height, 1.0 + directional_ratio, 1.0))
            append_variant(pred, scale_box_around_center(pred.box, width, height, 1.0, 1.0 + directional_ratio))
            append_variant(pred, scale_box_around_center(pred.box, width, height, max(0.01, 1.0 - directional_ratio), 1.0))
            append_variant(pred, scale_box_around_center(pred.box, width, height, 1.0, max(0.01, 1.0 - directional_ratio)))
        if max_new > 0 and len(additions) >= max_new:
            break

    if max_new > 0:
        additions = additions[:max_new]
    return preds + additions


def clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            out[labels == idx] = 1
    return out


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mask_iou(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> float:
    if mask_a is None or mask_b is None:
        return 0.0
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    return float(inter / union) if union > 0 else 0.0


def merge_predictions(preds: list[SegPrediction], box_iou_thr: float, mask_iou_thr: float) -> list[SegPrediction]:
    kept: list[SegPrediction] = []
    for pred in sorted(preds, key=lambda item: item.score, reverse=True):
        merged = False
        for kept_pred in kept:
            if box_iou(pred.box, kept_pred.box) < box_iou_thr:
                continue
            if pred.mask is not None and kept_pred.mask is not None and mask_iou(pred.mask, kept_pred.mask) < mask_iou_thr:
                continue
            if pred.mask is not None and kept_pred.mask is not None:
                kept_pred.mask = np.logical_or(kept_pred.mask > 0, pred.mask > 0).astype(np.uint8)
                new_box = mask_to_box(kept_pred.mask)
                if new_box is not None:
                    kept_pred.box = new_box
                    kept_pred.mask_box = new_box
            kept_pred.score = max(kept_pred.score, pred.score)
            merged = True
            break
        if not merged:
            kept.append(pred)
    return kept


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def select_tiles(windows: list[tuple[int, int]], max_tiles: int) -> list[tuple[int, int]]:
    if max_tiles <= 0 or len(windows) <= max_tiles:
        return windows
    if max_tiles == 1:
        return [windows[len(windows) // 2]]
    indices = np.linspace(0, len(windows) - 1, num=max_tiles, dtype=np.int32)
    return [windows[int(idx)] for idx in indices]


def predict_array(
    model: Any,
    image: np.ndarray,
    full_shape: tuple[int, int],
    offset: tuple[int, int],
    cfg: dict[str, Any],
    keep_full_mask: bool = True,
) -> list[SegPrediction]:
    infer_cfg = cfg["infer"]
    results = model.predict(
        image,
        imgsz=infer_cfg["imgsz"],
        conf=infer_cfg["conf"],
        iou=infer_cfg["iou"],
        max_det=infer_cfg["max_det"],
        verbose=False,
        retina_masks=bool(infer_cfg.get("retina_masks", True)),
        augment=bool(infer_cfg.get("augment", False)),
    )
    result = results[0]
    if result.boxes is None:
        return []

    out: list[SegPrediction] = []
    scores = result.boxes.conf.cpu().tolist()
    boxes = result.boxes.xyxy.cpu().tolist()
    masks = result.masks.data.cpu().numpy() if result.masks is not None else [None] * len(boxes)
    full_h, full_w = full_shape
    off_x, off_y = offset
    min_mask_area = int(infer_cfg.get("min_mask_area", 5))
    box_source = str(infer_cfg.get("prediction_box_source", "mask_box")).lower()
    if box_source not in {"mask_box", "det_box", "prefer_mask"}:
        raise ValueError("prediction_box_source must be one of: mask_box, det_box, prefer_mask")
    use_mask_boxes = box_source != "det_box" and bool(infer_cfg.get("retina_masks", True))

    for box, score, mask in zip(boxes, scores, masks):
        x1, y1, x2, y2 = box
        det_box = clip_xyxy([x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y], full_w, full_h)
        full_mask: np.ndarray | None = None
        if use_mask_boxes and mask is not None:
            mask_bin = clean_mask((mask > 0.5).astype(np.uint8), min_area=min_mask_area)
            if mask_bin.sum() >= min_mask_area:
                mh, mw = mask_bin.shape[:2]
                local_box = mask_to_box(mask_bin)
                if local_box is not None:
                    mask_box = [
                        local_box[0] + off_x,
                        local_box[1] + off_y,
                        local_box[2] + off_x,
                        local_box[3] + off_y,
                    ]
                    mask_box = clip_xyxy(mask_box, full_w, full_h)
                    if keep_full_mask:
                        full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
                        full_mask[off_y : off_y + mh, off_x : off_x + mw] = mask_bin
                    pred_box = det_box if box_source == "det_box" else mask_box
                    out.append(SegPrediction(box=pred_box, score=float(score), mask=full_mask, mask_box=mask_box, source="model"))
                    continue

        fallback_box = det_box
        out.append(
            SegPrediction(
                box=fallback_box,
                score=float(score),
                mask=full_mask,
                mask_box=mask_to_box(full_mask) if full_mask is not None else None,
                source="model",
            )
        )
    return out


def resize_for_global(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(width, height))
    if scale >= 1.0:
        return image, 1.0
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)), scale


def select_global_max_side(width: int, height: int, infer_cfg: dict[str, Any]) -> int:
    base = int(infer_cfg.get("global_max_side", infer_cfg["imgsz"]))
    adaptive = int(infer_cfg.get("adaptive_global_max_side", 0) or 0)
    threshold = int(infer_cfg.get("adaptive_global_threshold", 0) or 0)
    if adaptive > 0 and threshold > 0 and max(width, height) <= threshold:
        return adaptive
    return base


def predict_multiscale(model: Any, image: np.ndarray, cfg: dict[str, Any]) -> tuple[list[SegPrediction], float]:
    height, width = image.shape[:2]
    infer_cfg = cfg["infer"]
    preds: list[SegPrediction] = []
    start = time.perf_counter()

    keep_masks_for_merge = bool(infer_cfg.get("keep_masks_for_merge", True))

    if max(width, height) <= infer_cfg["direct_max_side"]:
        direct_resize_max_side = int(infer_cfg.get("direct_resize_max_side", 0) or 0)
        if direct_resize_max_side > 0 and max(width, height) > direct_resize_max_side:
            direct_img, scale = resize_for_global(image, direct_resize_max_side)
            for pred in predict_array(model, direct_img, direct_img.shape[:2], (0, 0), cfg, keep_full_mask=False):
                pred.box = [value / scale for value in pred.box]
                pred.box = clip_xyxy(pred.box, width, height)
                if pred.mask_box is not None:
                    pred.mask_box = clip_xyxy([value / scale for value in pred.mask_box], width, height)
                pred.mask = None
                preds.append(pred)
        else:
            preds.extend(predict_array(model, image, (height, width), (0, 0), cfg, keep_full_mask=keep_masks_for_merge))
    else:
        if infer_cfg.get("include_global_for_large", True):
            global_max_side = select_global_max_side(width, height, infer_cfg)
            global_img, scale = resize_for_global(image, global_max_side)
            for pred in predict_array(model, global_img, global_img.shape[:2], (0, 0), cfg, keep_full_mask=False):
                pred.box = [value / scale for value in pred.box]
                pred.box = clip_xyxy(pred.box, width, height)
                if pred.mask_box is not None:
                    pred.mask_box = clip_xyxy([value / scale for value in pred.mask_box], width, height)
                pred.mask = None
                preds.append(pred)

        tile = int(infer_cfg.get("tile_size", 0))
        trigger = str(infer_cfg.get("tile_trigger", "always")).lower()
        min_global_preds = int(infer_cfg.get("tile_min_global_preds", 1))
        should_tile = tile > 0 and (
            trigger == "always"
            or (trigger == "none" and False)
            or (trigger == "low_preds" and len(preds) < min_global_preds)
            or (trigger == "large" and max(width, height) > int(infer_cfg.get("direct_max_side", 2048)))
        )
        if should_tile:
            overlap = int(infer_cfg["tile_overlap"])
            windows = [(x0, y0) for y0 in tile_starts(height, tile, overlap) for x0 in tile_starts(width, tile, overlap)]
            windows = select_tiles(windows, int(infer_cfg.get("max_tiles", 0)))
            for x0, y0 in windows:
                crop = image[y0 : y0 + tile, x0 : x0 + tile]
                preds.extend(predict_array(model, crop, (height, width), (x0, y0), cfg, keep_full_mask=keep_masks_for_merge))

    preds = merge_predictions(
        preds,
        box_iou_thr=float(infer_cfg.get("box_iou_merge", 0.5)),
        mask_iou_thr=float(infer_cfg.get("mask_iou_merge", 0.35)),
    )
    expand_ratio = float(infer_cfg.get("box_expand_ratio", 0.0))
    expand_pixels = float(infer_cfg.get("box_expand_pixels", 0.0))
    expand_min_area = float(infer_cfg.get("box_expand_min_area", 0.0))
    expand_min_side = float(infer_cfg.get("box_expand_min_side", 0.0))
    if expand_ratio > 0 or expand_pixels > 0:
        for pred in preds:
            pred.box = expand_box(
                pred.box,
                width,
                height,
                ratio=expand_ratio,
                pixels=expand_pixels,
                min_area=expand_min_area,
                min_side=expand_min_side,
            )
    elong_ratio = float(infer_cfg.get("elongated_box_expand_ratio", 0.0))
    elong_pixels = float(infer_cfg.get("elongated_box_expand_pixels", 0.0))
    elong_min_area = float(infer_cfg.get("elongated_box_min_area", 0.0))
    elong_min_aspect = float(infer_cfg.get("elongated_box_min_aspect", 0.0))
    if elong_ratio > 0 or elong_pixels > 0:
        for pred in preds:
            pred.box = expand_elongated_box(
                pred.box,
                width,
                height,
                ratio=elong_ratio,
                pixels=elong_pixels,
                min_area=elong_min_area,
                min_aspect=elong_min_aspect,
            )
    edge_margin = float(infer_cfg.get("edge_anchor_expand_margin", 0.0))
    edge_min_area = float(infer_cfg.get("edge_anchor_min_area", 0.0))
    edge_min_aspect = float(infer_cfg.get("edge_anchor_min_aspect", 0.0))
    if edge_margin > 0:
        for pred in preds:
            pred.box = expand_edge_anchored_box(
                pred.box,
                width,
                height,
                min_area=edge_min_area,
                min_aspect=edge_min_aspect,
                edge_margin=edge_margin,
            )
    tiny_min_width = float(infer_cfg.get("tiny_box_min_width", 0.0))
    tiny_min_height = float(infer_cfg.get("tiny_box_min_height", 0.0))
    tiny_max_area = float(infer_cfg.get("tiny_box_max_area", 0.0))
    tiny_min_side = float(infer_cfg.get("tiny_box_min_side", 0.0))
    tiny_max_width = float(infer_cfg.get("tiny_box_max_width", 0.0))
    if tiny_min_width > 0 or tiny_min_height > 0:
        for pred in preds:
            pred.box = expand_tiny_box(
                pred.box,
                width,
                height,
                min_width=tiny_min_width,
                min_height=tiny_min_height,
                max_area=tiny_max_area,
                min_side=tiny_min_side,
                max_width=tiny_max_width,
            )
    preds = add_elongated_cluster_unions(preds, width, height, infer_cfg)
    preds = add_large_box_variants(preds, width, height, infer_cfg)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    preds.sort(key=lambda item: item.score, reverse=True)
    return preds, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO-seg inference and generate official bbox submission.")
    parser.add_argument("--config", default="configs/yolo_seg_crack.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit for the number of images.")
    parser.add_argument("--warmup", type=int, default=0, help="Run N warmup predictions before timed inference.")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--direct-max-side", type=int, default=None)
    parser.add_argument("--direct-resize-max-side", type=int, default=None)
    parser.add_argument("--global-max-side", type=int, default=None)
    parser.add_argument("--adaptive-global-max-side", type=int, default=None)
    parser.add_argument("--adaptive-global-threshold", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=None, help="Set 0 to disable tiled inference.")
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--tile-trigger", choices=["always", "none", "low_preds", "large"], default=None)
    parser.add_argument("--tile-min-global-preds", type=int, default=None)
    parser.add_argument("--max-tiles", type=int, default=None, help="0 means no tile limit.")
    parser.add_argument("--prediction-box-source", choices=["mask_box", "det_box", "prefer_mask"], default=None)
    parser.add_argument("--retina-masks", action="store_true", default=None)
    parser.add_argument("--no-retina-masks", action="store_false", dest="retina_masks")
    parser.add_argument("--box-expand-ratio", type=float, default=None)
    parser.add_argument("--box-expand-pixels", type=float, default=None)
    parser.add_argument("--box-expand-min-area", type=float, default=None)
    parser.add_argument("--box-expand-min-side", type=float, default=None)
    parser.add_argument("--tiny-box-min-width", type=float, default=None)
    parser.add_argument("--tiny-box-min-height", type=float, default=None)
    parser.add_argument("--tiny-box-max-area", type=float, default=None)
    parser.add_argument("--tiny-box-min-side", type=float, default=None)
    parser.add_argument("--tiny-box-max-width", type=float, default=None)
    parser.add_argument("--elongated-box-expand-ratio", type=float, default=None)
    parser.add_argument("--elongated-box-expand-pixels", type=float, default=None)
    parser.add_argument("--elongated-box-min-area", type=float, default=None)
    parser.add_argument("--elongated-box-min-aspect", type=float, default=None)
    parser.add_argument("--edge-anchor-expand-margin", type=float, default=None)
    parser.add_argument("--edge-anchor-min-area", type=float, default=None)
    parser.add_argument("--edge-anchor-min-aspect", type=float, default=None)
    parser.add_argument("--union-elongated-clusters", action="store_true", default=None)
    parser.add_argument("--no-union-elongated-clusters", action="store_false", dest="union_elongated_clusters")
    parser.add_argument("--union-cluster-min-area", type=float, default=None)
    parser.add_argument("--union-cluster-min-aspect", type=float, default=None)
    parser.add_argument("--union-cluster-max-gap", type=float, default=None)
    parser.add_argument("--union-cluster-min-cross-overlap", type=float, default=None)
    parser.add_argument("--union-cluster-min-members", type=int, default=None)
    parser.add_argument("--union-cluster-score-factor", type=float, default=None)
    parser.add_argument("--union-cluster-score-floor", type=float, default=None)
    parser.add_argument("--union-cluster-score-area-norm", type=float, default=None)
    parser.add_argument("--union-cluster-max-new-boxes", type=int, default=None)
    parser.add_argument("--union-cluster-box-source", choices=["box", "mask_box", "prefer_mask"], default=None)
    parser.add_argument("--large-box-variants", action="store_true", default=None)
    parser.add_argument("--no-large-box-variants", action="store_false", dest="large_box_variants")
    parser.add_argument("--large-variant-min-area", type=float, default=None)
    parser.add_argument("--large-variant-score-factor", type=float, default=None)
    parser.add_argument("--large-variant-score-floor", type=float, default=None)
    parser.add_argument("--large-variant-max-new-boxes", type=int, default=None)
    parser.add_argument("--large-variant-expand-ratio", type=float, default=None)
    parser.add_argument("--large-variant-shrink-ratio", type=float, default=None)
    parser.add_argument("--large-variant-directional-expand-ratio", type=float, default=None)
    parser.add_argument("--large-variant-min-delta", type=float, default=None)
    parser.add_argument("--large-variant-source-filter", choices=["all", "union", "model"], default=None)
    parser.add_argument("--include-global-for-large", action="store_true", default=None)
    parser.add_argument("--no-include-global-for-large", action="store_false", dest="include_global_for_large")
    parser.add_argument("--keep-masks-for-merge", action="store_true", default=None)
    parser.add_argument("--no-keep-masks-for-merge", action="store_false", dest="keep_masks_for_merge")
    parser.add_argument("--augment", action="store_true", default=None)
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    infer_cfg = cfg["infer"]
    overrides = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "direct_max_side": args.direct_max_side,
        "direct_resize_max_side": args.direct_resize_max_side,
        "global_max_side": args.global_max_side,
        "adaptive_global_max_side": args.adaptive_global_max_side,
        "adaptive_global_threshold": args.adaptive_global_threshold,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "tile_trigger": args.tile_trigger,
        "tile_min_global_preds": args.tile_min_global_preds,
        "max_tiles": args.max_tiles,
        "prediction_box_source": args.prediction_box_source,
        "retina_masks": args.retina_masks,
        "box_expand_ratio": args.box_expand_ratio,
        "box_expand_pixels": args.box_expand_pixels,
        "box_expand_min_area": args.box_expand_min_area,
        "box_expand_min_side": args.box_expand_min_side,
        "tiny_box_min_width": args.tiny_box_min_width,
        "tiny_box_min_height": args.tiny_box_min_height,
        "tiny_box_max_area": args.tiny_box_max_area,
        "tiny_box_min_side": args.tiny_box_min_side,
        "tiny_box_max_width": args.tiny_box_max_width,
        "elongated_box_expand_ratio": args.elongated_box_expand_ratio,
        "elongated_box_expand_pixels": args.elongated_box_expand_pixels,
        "elongated_box_min_area": args.elongated_box_min_area,
        "elongated_box_min_aspect": args.elongated_box_min_aspect,
        "edge_anchor_expand_margin": args.edge_anchor_expand_margin,
        "edge_anchor_min_area": args.edge_anchor_min_area,
        "edge_anchor_min_aspect": args.edge_anchor_min_aspect,
        "union_elongated_clusters": args.union_elongated_clusters,
        "union_cluster_min_area": args.union_cluster_min_area,
        "union_cluster_min_aspect": args.union_cluster_min_aspect,
        "union_cluster_max_gap": args.union_cluster_max_gap,
        "union_cluster_min_cross_overlap": args.union_cluster_min_cross_overlap,
        "union_cluster_min_members": args.union_cluster_min_members,
        "union_cluster_score_factor": args.union_cluster_score_factor,
        "union_cluster_score_floor": args.union_cluster_score_floor,
        "union_cluster_score_area_norm": args.union_cluster_score_area_norm,
        "union_cluster_max_new_boxes": args.union_cluster_max_new_boxes,
        "union_cluster_box_source": args.union_cluster_box_source,
        "large_box_variants": args.large_box_variants,
        "large_variant_min_area": args.large_variant_min_area,
        "large_variant_score_factor": args.large_variant_score_factor,
        "large_variant_score_floor": args.large_variant_score_floor,
        "large_variant_max_new_boxes": args.large_variant_max_new_boxes,
        "large_variant_expand_ratio": args.large_variant_expand_ratio,
        "large_variant_shrink_ratio": args.large_variant_shrink_ratio,
        "large_variant_directional_expand_ratio": args.large_variant_directional_expand_ratio,
        "large_variant_min_delta": args.large_variant_min_delta,
        "large_variant_source_filter": args.large_variant_source_filter,
        "include_global_for_large": args.include_global_for_large,
        "keep_masks_for_merge": args.keep_masks_for_merge,
        "augment": args.augment,
    }
    for key, value in overrides.items():
        if value is not None:
            infer_cfg[key] = value
    dataset_root = Path(cfg["dataset_root"])
    submissions_dir = Path(cfg["outputs"]["submissions"])
    submissions_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing ultralytics. Install dependencies from requirements.txt first.") from exc

    model = YOLO(args.weights)
    items, image_root = load_split_items(dataset_root, Path(cfg["prepared_root"]), args.split)
    if args.limit is not None:
        items = items[: args.limit]
    if args.warmup > 0 and items:
        warmup_image = load_rgb(image_root / items[0]["Image"])
        for _ in range(args.warmup):
            predict_multiscale(model, warmup_image, cfg)
        sync_cuda_if_available()
    results = []
    for idx, item in enumerate(items, start=1):
        img_path = image_root / item["Image"]
        image = load_rgb(img_path)
        height, width = image.shape[:2]
        preds, elapsed_ms = predict_multiscale(model, image, cfg)
        predict_bboxes = []
        for pred in preds:
            x1, y1, x2, y2 = round_box(clip_xyxy(pred.box, width, height))
            if x2 <= x1 or y2 <= y1:
                continue
            predict_bboxes.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": float(pred.score),
                    "label": "crack",
                }
            )
        results.append(
            {
                "ID": item["ID"],
                "image path": item["Image"],
                "inference_time_ms": round(elapsed_ms, 3),
                "groundtruth_bboxes": [],
                "predict_bboxes": predict_bboxes,
            }
        )
        if idx % 5 == 0 or idx == len(items):
            print(f"[{idx}/{len(items)}] processed", flush=True)

    out_path = Path(args.out) if args.out else submissions_dir / "results_seg.json"
    save_json(results, out_path)
    print(f"Saved YOLO-seg bbox submission to {out_path}")


if __name__ == "__main__":
    main()
