from __future__ import annotations

import random
import shutil
from pathlib import Path

RANDOM_SEED = 42

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def split_dataset(
    source_dir: Path,
    target_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(RANDOM_SEED)

    classes = [d for d in source_dir.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes")

    for class_dir in classes:
        class_name = class_dir.name
        images = sorted(class_dir.glob("*.*"))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split_name, split_files in splits.items():
            out_class_dir = target_dir / split_name / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)
            for src_path in split_files:
                dst_path = out_class_dir / src_path.name
                shutil.copy2(src_path, dst_path)

        print(
            f"{class_name}: total={n}, train={n_train}, "
            f"val={n_val}, test={n_test}"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_dir = project_root / "data" / "raw" / "fruit_ripeness" / "all"
    target_dir = project_root / "data" / "processed" / "fruit_ripeness"

    split_dataset(
        source_dir=source_dir,
        target_dir=target_dir,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )
    print("Done splitting dataset.")


if __name__ == "__main__":
    main()
