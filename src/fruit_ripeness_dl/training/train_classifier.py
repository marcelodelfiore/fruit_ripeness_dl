from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from fruit_ripeness_dl.config import TrainingConfig
from fruit_ripeness_dl.data.datasets import build_dataloaders
from fruit_ripeness_dl.models.cnn import FruitRipenessCNN
from fruit_ripeness_dl.training.loop import train_one_epoch, evaluate


def run_experiment(cfg: TrainingConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = build_dataloaders(cfg)

    num_classes = len(train_loader.dataset.classes)
    print(f"Detected {num_classes} classes")

    model = FruitRipenessCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    if cfg.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)

    runs_dir = Path(cfg.runs_dir)
    exp_dir = runs_dir / cfg.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"[{epoch+1}/{cfg.num_epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        history.append((epoch + 1, train_loss, val_loss, val_acc))

    hist_path = exp_dir / "history.csv"
    with hist_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
        writer.writerows(history)

    torch.save(model.state_dict(), exp_dir / "model_final.pth")
