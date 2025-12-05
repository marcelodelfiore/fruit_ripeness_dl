import argparse
import csv
import subprocess
from pathlib import Path
import shlex
from typing import Dict, List


# Later, if necessary, can accept args from command line, e.g.  "--device cuda --data-root data/processed".
TRAIN_BASE_ARGS = ""   # e.g. "--device cuda --data-root data/processed"
EVAL_BASE_ARGS = ""    # e.g. "--device cuda --data-root data/processed"


def project_root() -> Path:
    """Return the project root (assuming this file is in <root>/scripts/)."""
    return Path(__file__).resolve().parents[1]


def default_config_path(config_filename: str) -> Path:
    """
    Resolve config filename inside the default folder:
    <project_root>/configs/experiments/<config_filename>
    """
    return project_root() / "configs" / "experiments" / config_filename


def build_train_cmd(row: Dict[str, str]) -> List[str]:
    """
    Build the 'poetry run train ...' command from a CSV row.

    - 'experiment_name' (or 'run_name') is used as --experiment-name
    - 'epochs' column is mapped to --num-epochs
    - other columns become '--key value' flags, with '_' -> '-'
    """
    cmd: List[str] = ["poetry", "run", "train"]

    if TRAIN_BASE_ARGS:
        cmd.extend(shlex.split(TRAIN_BASE_ARGS))

    experiment_name = row.get("experiment_name") or row.get("run_name")

    for key, value in row.items():
        if not value:
            # skip empty cells
            continue

        if key in ("experiment_name", "run_name"):
            # handled separately as --experiment-name
            continue

        # Special case: 'epochs' -> '--num-epochs'
        if key == "epochs":
            flag = "--num-epochs"
        else:
            flag = "--" + key.replace("_", "-")

        cmd.extend([flag, str(value)])

    # Add experiment name if present
    if experiment_name:
        cmd.extend(["--experiment-name", experiment_name])

    return cmd


def build_eval_cmd(row: Dict[str, str]) -> List[str]:
    """
    Build the 'poetry run eval-test ...' command.

    We pass --experiment-name so evaluate_test.py loads the correct run.
    """
    cmd: List[str] = ["poetry", "run", "eval-test"]

    if EVAL_BASE_ARGS:
        cmd.extend(shlex.split(EVAL_BASE_ARGS))

    experiment_name = row.get("experiment_name") or row.get("run_name")
    if experiment_name:
        cmd.extend(["--experiment-name", experiment_name])

    return cmd


def run_batch_from_csv(config_filename: str) -> None:
    config_path = default_config_path(config_filename)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"[batch_experiment] Using config file: {config_path}")

    successes: list[str] = []
    failures: list[str] = []

    with config_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "experiment_name" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain an 'experiment_name' column.")

        for idx, row in enumerate(reader, start=1):
            experiment_name = row.get("experiment_name") or f"experiment_{idx}"

            print(f"\n=== Running experiment {idx}: {experiment_name} ===")

            # 1) Train
            train_cmd = build_train_cmd(row)
            print("[train] Command:", " ".join(train_cmd))
            try:
                subprocess.run(train_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[train] Experiment '{experiment_name}' failed with exit code {e.returncode}")
                failures.append(experiment_name)
                continue  # move on to the next row

            # 2) Evaluate
            eval_cmd = build_eval_cmd(row)
            print("[eval-test] Command:", " ".join(eval_cmd))
            try:
                subprocess.run(eval_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[eval-test] Experiment '{experiment_name}' failed with exit code {e.returncode}")
                failures.append(experiment_name)
                continue

            print(f"[OK] Experiment '{experiment_name}' completed.")
            successes.append(experiment_name)

    print("\n=== Summary ===")
    if successes:
        print("Successful experiments:")
        for name in successes:
            print("  -", name)
    else:
        print("No successful experiments.")

    if failures:
        print("\nFailed experiments:")
        for name in failures:
            print("  -", name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple train + eval-test experiments from a CSV config file."
    )
    parser.add_argument(
        "config_filename",
        help=(
            "Name of the CSV file inside configs/experiments/. "
            "Example: variations.csv"
        ),
    )
    args = parser.parse_args()
    run_batch_from_csv(args.config_filename)


if __name__ == "__main__":
    main()
