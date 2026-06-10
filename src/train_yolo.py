from __future__ import annotations

import argparse
from pathlib import Path

from common import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO detector for CPIPC crack detection.")
    parser.add_argument("--config", default="configs/yolo_crack.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg["train"]
    prepared_root = Path(cfg["prepared_root"])
    data_yaml = prepared_root / "crack.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found. Run src/prepare_yolo.py first.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing ultralytics. Install dependencies from requirements.txt first.") from exc

    model_name = args.model or train_cfg["model"]
    model = YOLO(model_name)
    device = args.device or train_cfg.get("device", "auto")
    if device == "auto":
        device = None

    project = Path(args.project or train_cfg["project"]).resolve()
    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz or train_cfg["imgsz"],
        epochs=args.epochs or train_cfg["epochs"],
        batch=args.batch or train_cfg["batch"],
        device=device,
        project=str(project),
        name=args.name or train_cfg["name"],
        workers=train_cfg.get("workers", 4),
        patience=train_cfg.get("patience", 30),
        close_mosaic=train_cfg.get("close_mosaic", 10),
        exist_ok=True,
        single_cls=True,
        plots=True,
        cache=False,
        degrees=90,
        translate=0.08,
        scale=0.4,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
    )


if __name__ == "__main__":
    main()
