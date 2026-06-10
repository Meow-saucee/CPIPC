from __future__ import annotations

import argparse
from pathlib import Path

from common import load_yaml
from experiment_utils import archive_experiment, build_experiment_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive one YOLO run with ckpts, metrics and TensorBoard logs.")
    parser.add_argument("--run-dir", required=True, help="Ultralytics run directory, e.g. runs/crack_yolo/train")
    parser.add_argument("--config", default="configs/yolo_crack.yaml")
    parser.add_argument("--exp-root", default=None)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--errors", default=None)
    parser.add_argument("--submission", default=None)
    parser.add_argument("--command", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg.get("train", {})
    split_cfg = cfg.get("split", {})
    exp_cfg = cfg.get("experiment", {})

    model = args.model or train_cfg.get("model", "model")
    dataset_name = args.dataset_name or exp_cfg.get("dataset_name", "dataset")
    imgsz = args.imgsz or train_cfg.get("imgsz")
    epochs = args.epochs or train_cfg.get("epochs")
    batch = args.batch or train_cfg.get("batch")
    seed = args.seed if args.seed is not None else split_cfg.get("seed")
    tag = args.tag if args.tag is not None else exp_cfg.get("tag")
    exp_name = args.exp_name or build_experiment_name(
        model=model,
        dataset_name=dataset_name,
        imgsz=int(imgsz),
        epochs=int(epochs),
        batch=int(batch),
        seed=seed,
        tag=tag,
    )
    exp_root = Path(args.exp_root or exp_cfg.get("root", "experiments"))

    extra_files: dict[str, Path] = {}
    if args.metrics:
        extra_files["metrics"] = Path(args.metrics)
    if args.errors:
        extra_files["errors"] = Path(args.errors)
    if args.submission:
        extra_files["submission"] = Path(args.submission)

    metadata = {
        "task": "cpipc_crack_detection",
        "model": model,
        "dataset_name": dataset_name,
        "params": {
            "imgsz": imgsz,
            "epochs": epochs,
            "batch": batch,
            "seed": seed,
            "tag": tag,
        },
        "config": args.config,
        "command": args.command,
    }
    exp_dir = archive_experiment(
        run_dir=Path(args.run_dir),
        exp_root=exp_root,
        exp_name=exp_name,
        metadata=metadata,
        extra_files=extra_files,
    )
    print(f"Archived experiment to {exp_dir}")


if __name__ == "__main__":
    main()
