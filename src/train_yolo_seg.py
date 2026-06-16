from __future__ import annotations

import argparse
from pathlib import Path

from common import load_yaml
from experiment_utils import archive_experiment, build_experiment_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model for CPIPC crack detection.")
    parser.add_argument("--config", default="configs/yolo_seg_crack.yaml")
    parser.add_argument("--data-yaml", default=None, help="Override YOLO data yaml, e.g. data/yolo_seg/crack_seg_scaleaware.yaml.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--exp-root", default=None)
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg["train"]
    split_cfg = cfg.get("split", {})
    exp_cfg = cfg.get("experiment", {})
    data_yaml = Path(args.data_yaml) if args.data_yaml else Path(cfg["prepared_root"]) / "crack_seg.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found. Run src/prepare_yolo_seg.py first.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing ultralytics. Install dependencies from requirements.txt first.") from exc

    model_name = args.model or train_cfg["model"]
    model = YOLO(model_name)
    device = args.device or train_cfg.get("device", "auto")
    if device == "auto":
        device = None

    imgsz = args.imgsz or train_cfg["imgsz"]
    epochs = args.epochs or train_cfg["epochs"]
    batch = args.batch or train_cfg["batch"]
    seed = args.seed if args.seed is not None else split_cfg.get("seed")
    dataset_name = args.dataset_name or exp_cfg.get("dataset_name", "dataset")
    tag = args.tag if args.tag is not None else exp_cfg.get("tag")
    name = args.name or train_cfg["name"]
    if name == "auto":
        name = build_experiment_name(model_name, dataset_name, imgsz, epochs, batch, seed=seed, tag=tag)

    project = Path(args.project or train_cfg["project"]).resolve()
    model.train(
        data=str(data_yaml),
        task="segment",
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        workers=train_cfg.get("workers", 4),
        patience=train_cfg.get("patience", 50),
        close_mosaic=train_cfg.get("close_mosaic", 20),
        optimizer=train_cfg.get("optimizer", "auto"),
        mask_ratio=train_cfg.get("mask_ratio", 4),
        exist_ok=True,
        single_cls=True,
        overlap_mask=True,
        plots=True,
        cache=False,
        degrees=90,
        translate=0.06,
        scale=0.35,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.8,
        mixup=0.03,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
    )

    if exp_cfg.get("archive_after_train", True) and not args.no_archive:
        exp_root = Path(args.exp_root or exp_cfg.get("root", "experiments"))
        run_dir = project / name
        exp_dir = archive_experiment(
            run_dir=run_dir,
            exp_root=exp_root,
            exp_name=name,
            metadata={
                "task": "cpipc_crack_instance_segmentation",
                "model": model_name,
                "dataset_name": dataset_name,
                "params": {
                    "imgsz": imgsz,
                    "epochs": epochs,
                    "batch": batch,
                    "seed": seed,
                    "tag": tag,
                    "workers": train_cfg.get("workers", 4),
                    "patience": train_cfg.get("patience", 50),
                    "close_mosaic": train_cfg.get("close_mosaic", 20),
                    "mask_ratio": train_cfg.get("mask_ratio", 4),
                },
                "config": args.config,
                "data_yaml": str(data_yaml),
            },
        )
        print(f"Archived experiment to {exp_dir}")


if __name__ == "__main__":
    main()
