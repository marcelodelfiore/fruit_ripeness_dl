from __future__ import annotations

import argparse
import time

from fruit_ripeness_dl.config import TrainingConfig
from fruit_ripeness_dl.training.train_classifier import run_experiment as _run_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the fruit ripeness classifier."
    )

    parser.add_argument(
        "--experiment-name",
        "-e",
        default="exp_baseline",
        help="Name of the experiment / run directory.",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        "-l",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--num-epochs",
        "-n",
        type=int,
        default=10,
        help="Number of training epochs (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=None,
        help="Override batch size (if not set, uses TrainingConfig default).",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="Override device (e.g. 'cuda', 'cpu'). If not set, uses TrainingConfig default.",
    )
    parser.add_argument(
        "--optimizer",
        "-o",
        default=None,
        help="Override optimizer (e.g. 'adam', 'sgd'). If not set, uses TrainingConfig default.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = TrainingConfig(
        experiment_name=args.experiment_name,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        optimizer=args.optimizer,
    )

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    if args.device is not None:
        cfg.device = args.device

    print("=== Training configuration ===")
    print(f"  experiment_name: {cfg.experiment_name}")
    print(f"  learning_rate:   {cfg.learning_rate}")
    print(f"  num_epochs:      {cfg.num_epochs}")
    if hasattr(cfg, "batch_size"):
        print(f"  batch_size:      {getattr(cfg, 'batch_size')}")
    if hasattr(cfg, "device"):
        print(f"  device:          {getattr(cfg, 'device')}")
    if hasattr(cfg, "optimizer"):
        print(f"  optimizer:       {getattr(cfg, 'optimizer')}")
    print("==============================")

    start = time.perf_counter()
    _run_experiment(cfg)
    elapsed = time.perf_counter() - start

    minutes, seconds = divmod(elapsed, 60)
    print(
        f"[train] Finished in {int(minutes)}m {seconds:04.1f}s "
        f"({elapsed:.2f} s total)."
    )


if __name__ == "__main__":
    main()
