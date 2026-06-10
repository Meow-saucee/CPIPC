from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def slugify(value: str) -> str:
    value = Path(value).stem if value.endswith((".pt", ".pth")) else value
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    value = re.sub(r"-+", "-", value).strip("-_.")
    return value or "exp"


def build_experiment_name(
    model: str,
    dataset_name: str,
    imgsz: int,
    epochs: int,
    batch: int,
    seed: int | None = None,
    tag: str | None = None,
    timestamp: str | None = None,
) -> str:
    parts = [
        slugify(model),
        slugify(dataset_name),
        f"img{imgsz}",
        f"ep{epochs}",
        f"bs{batch}",
    ]
    if seed is not None:
        parts.append(f"seed{seed}")
    if tag:
        parts.append(slugify(tag))
    if timestamp:
        parts.append(timestamp)
    return "_".join(parts)


def read_results(results_csv: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                key = key.strip()
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if "epoch" in parsed:
                rows.append(parsed)
    return rows


def summarize_results(results_csv: Path) -> dict[str, Any]:
    rows = read_results(results_csv)
    if not rows:
        raise ValueError(f"No numeric rows found in {results_csv}")

    last = rows[-1]
    best = max(rows, key=lambda item: item.get("metrics/mAP50(B)", float("-inf")))
    return {
        "last_epoch": int(last["epoch"]),
        "best_epoch": int(best["epoch"]),
        "best_metric": "metrics/mAP50(B)",
        "best_metrics": {
            "precision": best.get("metrics/precision(B)"),
            "recall": best.get("metrics/recall(B)"),
            "mAP50": best.get("metrics/mAP50(B)"),
            "mAP50_95": best.get("metrics/mAP50-95(B)"),
        },
        "last_metrics": {
            "precision": last.get("metrics/precision(B)"),
            "recall": last.get("metrics/recall(B)"),
            "mAP50": last.get("metrics/mAP50(B)"),
            "mAP50_95": last.get("metrics/mAP50-95(B)"),
        },
    }


def write_tensorboard_from_results(results_csv: Path, log_dir: Path) -> bool:
    rows = read_results(results_csv)
    if not rows:
        return False
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return False

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    for row in rows:
        step = int(row["epoch"])
        for key, value in row.items():
            if key == "epoch":
                continue
            writer.add_scalar(key, value, step)
    writer.flush()
    writer.close()
    return True


def copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def archive_experiment(
    run_dir: Path,
    exp_root: Path,
    exp_name: str,
    metadata: dict[str, Any],
    extra_files: dict[str, Path] | None = None,
) -> Path:
    run_dir = run_dir.resolve()
    exp_dir = (exp_root / exp_name).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "reports").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)

    results_csv = run_dir / "results.csv"
    summary = summarize_results(results_csv)
    metadata = {
        **metadata,
        "experiment_name": exp_name,
        "archived_at": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(run_dir),
        **summary,
    }

    best_epoch = metadata["best_epoch"]
    last_epoch = metadata["last_epoch"]
    best_map50 = metadata["best_metrics"]["mAP50"]
    ckpt_prefix = exp_name
    metadata["checkpoints"] = {
        "best": copy_if_exists(
            run_dir / "weights" / "best.pt",
            exp_dir / "checkpoints" / f"{ckpt_prefix}__best_epoch{best_epoch}_mAP50-{best_map50:.4f}.pt",
        ),
        "last": copy_if_exists(
            run_dir / "weights" / "last.pt",
            exp_dir / "checkpoints" / f"{ckpt_prefix}__last_epoch{last_epoch}.pt",
        ),
    }

    metadata["reports"] = {
        "results_csv": copy_if_exists(results_csv, exp_dir / "reports" / "results.csv"),
        "args_yaml": copy_if_exists(run_dir / "args.yaml", exp_dir / "reports" / "args.yaml"),
    }

    for plot in run_dir.glob("*.png"):
        copy_if_exists(plot, exp_dir / "plots" / plot.name)
    for image in run_dir.glob("*.jpg"):
        copy_if_exists(image, exp_dir / "plots" / image.name)

    if extra_files:
        metadata["extra_files"] = {}
        for name, path in extra_files.items():
            copied = copy_if_exists(path, exp_dir / "reports" / path.name)
            metadata["extra_files"][name] = copied

    metadata["tensorboard_logdir"] = str(exp_dir / "tensorboard")
    metadata["tensorboard_events_created"] = write_tensorboard_from_results(
        results_csv,
        exp_dir / "tensorboard",
    )

    with (exp_dir / "experiment.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with (exp_dir / "experiment.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True)

    return exp_dir
