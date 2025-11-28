from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import torch

from fruit_ripeness_dl.config import TrainingConfig
from fruit_ripeness_dl.data.datasets import build_dataloaders


def _to_int_list(values: Any) -> list[int]:
    """
    Normalize various target formats (list, tensor, etc.) to a list of ints.
    """
    if isinstance(values, torch.Tensor):
        values = values.tolist()

    if isinstance(values, (list, tuple)):
        return [int(v) for v in values]

    return [int(values)]


def _get_targets(dataset: Any) -> list[int]:
    """
    Try to extract labels/targets from common dataset attributes.
    Fallback: iterate over the dataset.
    """
    if hasattr(dataset, "targets"):
        return _to_int_list(dataset.targets)
    if hasattr(dataset, "labels"):
        return _to_int_list(dataset.labels)

    targets: list[int] = []
    for _, y in dataset:
        if isinstance(y, torch.Tensor):
            y = y.item()
        targets.append(int(y))
    return targets


def _print_split_summary(name: str, dataset: Any, class_names: list[str]) -> None:
    targets = _get_targets(dataset)
    counts = Counter(targets)
    total = len(targets)

    print(f"\n{name} split:")
    print(f"  Total samples: {total}")

    for idx, class_name in enumerate(class_names):
        count = counts.get(idx, 0)
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"  - {class_name:20s}: {count:5d} ({pct:5.1f}%)")


def main() -> None:
    cfg = TrainingConfig(
        experiment_name="exp_baseline",
    )

    print("Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    test_ds = test_loader.dataset

    class_names = list(train_ds.classes)
    num_classes = len(class_names)

    print("\n=== Dataset summary ===")
    print(f"Detected {num_classes} classes:")
    for i, name in enumerate(class_names):
        print(f"  [{i}] {name}")

    for batch in train_loader:
        images, labels = batch
        print("\nExample batch shapes:")
        print(f"  images: {tuple(images.shape)}")
        print(f"  labels: {tuple(labels.shape)}")
        break

    _print_split_summary("Train", train_ds, class_names)
    _print_split_summary("Val",   val_ds,   class_names)
    _print_split_summary("Test",  test_ds,  class_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
