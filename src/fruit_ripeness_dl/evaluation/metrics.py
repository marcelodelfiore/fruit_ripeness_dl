from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on dataloader and return (y_true, y_pred) as numpy arrays."""
    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []

    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    return y_true, y_pred


def compute_confusion_and_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> Tuple[np.ndarray, str]:
    """Return confusion matrix and a text classification report."""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    return cm, report
