from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fruit_ripeness_dl.config import TrainingConfig
from fruit_ripeness_dl.data.datasets import build_dataloaders
from fruit_ripeness_dl.evaluation.metrics import (
    collect_predictions,
    compute_confusion_and_report,
)
from fruit_ripeness_dl.models.cnn import FruitRipenessCNN


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained experiment on the test set."
    )
    parser.add_argument(
        "--experiment-name",
        "-e",
        default="exp_baseline",
        help="Name of the experiment to evaluate (default: exp_baseline).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = TrainingConfig(
        experiment_name=args.experiment_name,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    num_classes = len(train_loader.dataset.classes)
    class_names = train_loader.dataset.classes
    print(f"Detected {num_classes} classes")

    model = FruitRipenessCNN(num_classes=num_classes).to(device)

    runs_dir = Path(cfg.runs_dir)
    exp_dir = runs_dir / cfg.experiment_name
    model_path = exp_dir / "model_final.pth"

    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    y_true, y_pred = collect_predictions(model, test_loader, device)

    cm, report = compute_confusion_and_report(y_true, y_pred, class_names)

    print("Classification report (test):")
    print(report)

    # Save confusion matrix to disk for notebook
    cm_path = exp_dir / "confusion_matrix.npy"
    np.save(cm_path, cm)
    print(f"Saved confusion matrix to {cm_path}")

    classes_path = exp_dir / "class_names.txt"
    with classes_path.open("w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"Saved class names to {classes_path}")


if __name__ == "__main__":
    main()
