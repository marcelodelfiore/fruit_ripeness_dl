from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _prepare_output_dirs(root: Path, overwrite: bool) -> None:
    """Create empty train/val/test dirs under `root`."""
    for split in ("train", "val", "test"):
        split_dir = root / split
        if split_dir.exists():
            if not overwrite:
                raise SystemExit(
                    f"Output directory {split_dir} already exists. "
                    "Use --overwrite if you want to recreate the splits."
                )
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def _maybe_run_fix_names_script(script_path: Path, target_dir: Path, enabled: bool) -> None:
    """Optionally run a bash script to fix whitespaces in filenames."""
    if not enabled:
        return

    if not script_path.is_file():
        print(f"[WARN] Fix-names script not found at {script_path}, skipping.")
        return

    print(f"[INFO] Running filename-fix script: {script_path}")
    # If your script expects the target dir as argument, keep the second arg.
    # If not, you can remove `str(target_dir)` below.
    subprocess.run(["bash", str(script_path), str(target_dir)], check=True)


def _unzip_dataset(zip_path: Path, extract_root: Path, overwrite: bool) -> Path:
    """
    Unzip the dataset into `extract_root`.

    Returns the directory that actually contains the class folders.
    """
    if extract_root.exists():
        if not overwrite:
            raise SystemExit(
                f"Extract root {extract_root} already exists. "
                "Use --overwrite to re-extract."
            )
        shutil.rmtree(extract_root)

    extract_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Unzipping {zip_path} -> {extract_root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    candidates = [p for p in extract_root.iterdir() if p.is_dir()]
    has_images_here = any(_is_image_file(p) for p in extract_root.iterdir())

    if len(candidates) == 1 and not has_images_here:
        dataset_root = candidates[0]
        print(f"[INFO] Using inner directory as dataset root: {dataset_root}")
    else:
        dataset_root = extract_root
        print(f"[INFO] Using extract root as dataset root: {dataset_root}")

    return dataset_root


def _collect_class_dirs(dataset_root: Path) -> list[Path]:
    """
    Collect class directories directly under dataset_root.

    Assumes structure like:
      dataset_root/
        class_1/
          img1.jpg
        class_2/
          img2.jpg
    """
    class_dirs = [p for p in dataset_root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise SystemExit(f"No class directories found under {dataset_root}")
    return class_dirs


def _split_files(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    random.Random(seed).shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


def _copy_split(
    split_name: str,
    files: Iterable[Path],
    class_name: str,
    output_root: Path,
) -> None:
    dest_dir = output_root / split_name / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for src in files:
        dest = dest_dir / src.name
        shutil.copy2(src, dest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare dataset splits from a Kaggle zip.\n\n"
            "Assumes the zip is at data/raw/fruit_ripeness.zip and will create\n"
            "data/processed/train|val|test/<class_name>/..."
        )
    )

    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("data/raw/fruit_ripeness.zip"),
        help="Path to the Kaggle zip file (default: data/raw/fruit_ripeness.zip).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Root directory for processed splits (default: data/processed).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15). "
        "Test ratio is computed as 1 - train - val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42).",
    )
    parser.add_argument(
        "--fix-names-script",
        type=Path,
        default=Path("bash_scripts/fix_whitespace_filenames.sh"),
        help=(
            "Path to the bash script that fixes whitespace in file/folder names. "
            "Default: bash_scripts/fix_whitespace_filenames.sh"
        ),
    )
    parser.add_argument(
        "--run-fix-names",
        action="store_true",
        help="If set, run the fix-names bash script after unzipping.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing extracted / processed dirs.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.zip_path.is_file():
        raise SystemExit(f"Zip file not found: {args.zip_path}")

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise SystemExit(
            f"Invalid ratios: train={args.train_ratio}, val={args.val_ratio}. "
            "Ensure train + val < 1.0."
        )

    print("[INFO] Configuration:")
    print(f"  zip_path:     {args.zip_path}")
    print(f"  output_root:  {args.output_root}")
    print(f"  train_ratio:  {args.train_ratio}")
    print(f"  val_ratio:    {args.val_ratio}")
    print(f"  test_ratio:   {test_ratio}")
    print(f"  seed:         {args.seed}")
    print(f"  overwrite:    {args.overwrite}")
    if args.run_fix_names:
        print(f"  fix script:   {args.fix_names_script}")
    print("")

    # 1) Unzip
    extract_root = args.zip_path.with_suffix("")  # e.g. fruit_ripeness.zip -> fruit_ripeness
    dataset_root = _unzip_dataset(args.zip_path, extract_root, overwrite=args.overwrite)

    # 2) Optionally fix whitespace in file/folder names
    _maybe_run_fix_names_script(
        script_path=args.fix_names_script,
        target_dir=dataset_root,
        enabled=args.run_fix_names,
    )

    # 3) Prepare output dirs
    _prepare_output_dirs(args.output_root, overwrite=args.overwrite)

    # 4) Collect class dirs and split
    class_dirs = _collect_class_dirs(dataset_root)
    print(f"[INFO] Found {len(class_dirs)} class directories.")

    for class_dir in class_dirs:
        class_name = class_dir.name
        all_images = [p for p in class_dir.rglob("*") if _is_image_file(p)]

        if not all_images:
            print(f"[WARN] No images found in class dir {class_dir}, skipping.")
            continue

        splits = _split_files(
            files=all_images,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        print(
            f"[INFO] Class '{class_name}': "
            f"{len(all_images)} images -> "
            f"train={len(splits['train'])}, "
            f"val={len(splits['val'])}, "
            f"test={len(splits['test'])}"
        )

        for split_name, files in splits.items():
            _copy_split(split_name, files, class_name, args.output_root)

    print("\n[INFO] Done. Splits created under:")
    print(f"  {args.output_root / 'train'}")
    print(f"  {args.output_root / 'val'}")
    print(f"  {args.output_root / 'test'}")


if __name__ == "__main__":
    main()
