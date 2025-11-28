# Fruit Ripeness Classification (CNN + PyTorch)

This repository contains a complete deep learning pipeline to classify **fruit ripeness** using a Convolutional Neural Network (CNN) built with **PyTorch**.

The project includes:

- Scripts to **prepare and split the dataset**
- A configurable **training pipeline**
- Tools to **run experiments**, **evaluate on the test set**, and **summarize the dataset**

Everything is wired through **Poetry shortcuts**, so common tasks are easy and consistent to run.

---

## Requirements

- **Python**: 3.12+ (or another supported 3.x, as configured in `pyproject.toml`)
- **Poetry**: dependency and environment manager

Install Poetry (if you don't have it):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

---

## Installation

Clone this repository and install dependencies with Poetry:

```bash
git clone https://github.com/<your-username>/fruit_ripeness_dl.git
cd fruit_ripeness_dl

# Install dependencies
poetry install
```

You can optionally open a shell inside the virtual environment:

```bash
poetry shell
```

If you don’t want to use `poetry shell`, just prefix all commands with `poetry run`, e.g.:

```bash
poetry run python -m pip list
```

---

## Dataset

This project uses the following Kaggle dataset:

> **Fruit Image Dataset (22 Classes)**
> https://www.kaggle.com/datasets/mdsagorahmed/fruit-image-dataset-22-classes

### Download and place the dataset

1. Download the `.zip` file from Kaggle.
2. Create the following folder structure inside the repo:

```text
fruit_ripeness_dl/
  data/
    raw/
      fruit_ripeness.zip
```

The important part is that the downloaded zip is saved as:

```text
data/raw/fruit_ripeness.zip
```

(If Kaggle gives you a different name, just rename it to `fruit_ripeness.zip`.)

---

## Command shortcuts (Poetry scripts)

Common tasks are registered as **Poetry scripts** in `pyproject.toml`:

```toml
[tool.poetry.scripts]
model-split    = "scripts.model_split:main"
run-experiment = "scripts.run_experiment:main"
eval-test      = "scripts.evaluate_test:main"
train          = "scripts.train:main"
data-summary   = "scripts.data_summary:main"
```

You run them like this (from the project root):

```bash
poetry run model-split
poetry run train
poetry run run-experiment
poetry run eval-test
poetry run data-summary
```

---

## Typical workflow (for newcomers)

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/fruit_ripeness_dl.git
cd fruit_ripeness_dl
poetry install
```

### 2. Download dataset from Kaggle

Download the dataset from Kaggle and place the zip file at:

```text
data/raw/fruit_ripeness.zip
```

### 3. Prepare data splits

```bash
poetry run model-split --run-fix-names --overwrite
```

This will:

- Unzip `data/raw/fruit_ripeness.zip`
- Optionally fix whitespace in filenames using your bash script (because of `--run-fix-names`)
- Create `train`, `val`, and `test` splits under `data/processed/`

### 4. Inspect dataset

```bash
poetry run data-summary
```

This prints:

- Number of classes
- Class names
- Example batch shapes
- Samples per class in train/val/test (counts and percentages)

### 5. Train a baseline model

```bash
poetry run train   --experiment-name exp_baseline   --learning-rate 1e-3   --num-epochs 10   --batch-size 32   --device cuda
```

This trains a CNN and saves artifacts under `runs/exp_baseline/`.

### 6. Evaluate on the test split

```bash
poetry run eval-test
```

This loads the trained model (by default, for `exp_baseline`) and evaluates it on the test set, printing a classification report and saving the confusion matrix.

### 7. (Optional) Run the hard-coded baseline experiment wrapper

```bash
poetry run run-experiment
```

This is a simple wrapper that runs a predefined configuration (baseline experiment) without CLI arguments.

---

## Preparing and splitting the dataset

The main command for splitting is:

```bash
poetry run model-split
```

By default, it:

- Uses `data/raw/fruit_ripeness.zip`
- Extracts to `data/raw/fruit_ripeness/` (or similar)
- Creates processed splits under `data/processed/`:

  ```text
  data/processed/
    train/<class_name>/...
    val/<class_name>/...
    test/<class_name>/...
  ```

- Uses split ratios:
  - `train`: 0.70
  - `val`: 0.15
  - `test`: 0.15
- Uses random seed `42` for reproducible splits

### Model split – full example with options

```bash
poetry run model-split   --zip-path data/raw/fruit_ripeness.zip   --output-root data/processed   --train-ratio 0.75   --val-ratio 0.15   --seed 123   --run-fix-names   --overwrite
```

**Options:**

- `--zip-path PATH`  
  Path to the Kaggle zip file  
  *Default*: `data/raw/fruit_ripeness.zip`

- `--output-root PATH`  
  Root directory for processed splits  
  *Default*: `data/processed`

- `--train-ratio FLOAT`  
  Train split ratio (e.g., `0.75`)  
  *Default*: `0.7`

- `--val-ratio FLOAT`  
  Validation split ratio (e.g., `0.15`)  
  Test split is computed as `1 - train_ratio - val_ratio`.  
  *Default*: `0.15`

- `--seed INT`  
  Random seed for splitting  
  *Default*: `42`

- `--run-fix-names`  
  If provided, runs the whitespace-fix bash script after unzipping  
  (Default script path: `bash_scripts/fix_whitespace_filenames.sh`)

- `--fix-names-script PATH`  
  Custom path to the bash script that fixes filenames

- `--overwrite`  
  If provided, removes existing extracted/split directories before recreating them

---

## Training the model

The main flexible entrypoint for training is:

```bash
poetry run train
```

This uses `TrainingConfig` (from `fruit_ripeness_dl.config`) and the `run_experiment` function under the hood.

### Training options

You can customize training via CLI arguments:

```bash
poetry run train   --experiment-name exp_baseline   --learning-rate 1e-3   --num-epochs 10   --batch-size 32   --device cuda
```

Supported arguments:

- `--experiment-name, -e`  
  Name of the experiment (used to create `runs/<experiment_name>/`)

- `--learning-rate, --lr, -l`  
  Learning rate  
  *Default*: `1e-3`

- `--num-epochs, -n`  
  Number of training epochs  
  *Default*: `10`

- `--batch-size, -b`  
  Override batch size  
  *(If omitted, the default from `TrainingConfig` is used.)*

- `--device, -d`  
  Device for training, e.g.:
  - `cuda` (GPU)
  - `cuda:0`
  - `cpu`  
  *(If omitted, the value from `TrainingConfig` is used.)*

---

## Where training artifacts are stored

Training artifacts are stored under a `runs/` directory, with one subfolder per experiment:

```text
runs/
  exp_baseline/
    model_final.pth
    confusion_matrix.npy      # created by eval-test
    class_names.txt           # created by eval-test
    # (optionally) logs, metrics, etc.
```

- `model_final.pth` – final model checkpoint saved by the training pipeline.
- `confusion_matrix.npy` – NumPy array of the confusion matrix (saved by `eval-test`).
- `class_names.txt` – list of class names used in the model (saved by `eval-test`).

---

## Running experiments (simple wrapper)

In addition to the flexible `train` command, there is a convenience command:

```bash
poetry run run-experiment
```

This calls `run_experiment` with a hardcoded `TrainingConfig`, typically something like:

```python
cfg = TrainingConfig(
    experiment_name="exp_baseline",
    learning_rate=1e-3,
    num_epochs=10,
)
run_experiment(cfg)
```

Use this when you just want to re-run the baseline experiment with its default settings, without passing CLI arguments.

---

## Evaluating on the test set

Once you have trained a model (e.g., `exp_baseline`), you can evaluate it on the test set with:

```bash
poetry run eval-test
```

This script:

1. Creates the dataloaders (train/val/test) using `TrainingConfig`.
2. Infers the number of classes from the dataset.
3. Loads the trained model checkpoint for the configured experiment (`model_final.pth`).
4. Runs predictions on the **test** split only.
5. Computes:
   - Confusion matrix
   - Classification report (precision, recall, F1-score, support)
6. Prints the classification report to the console.
7. Saves:
   - `confusion_matrix.npy` in `runs/<experiment_name>/`
   - `class_names.txt` in `runs/<experiment_name>/`

By default, the script uses:

```python
cfg = TrainingConfig(
    experiment_name="exp_baseline",
)
```

So it expects:

```text
runs/exp_baseline/model_final.pth
```

If you train with a different `--experiment-name`, you should adjust the `experiment_name` in `evaluate_test.py` (or extend it to accept a `--experiment-name` argument).

---

## Dataset summary / sanity check

Before or after training, you can inspect the dataset using:

```bash
poetry run data-summary
```

This command:

1. Builds train/val/test dataloaders using `TrainingConfig`.
2. Prints:
   - Number of classes
   - Class names
3. Shows example batch shapes (e.g. `(batch_size, channels, height, width)`).
4. For each split (`train`, `val`, `test`), prints:
   - Total number of samples
   - Per-class counts and percentages

Example (structure):

```text
=== Dataset summary ===
Detected 22 classes:
  [0] ripe_apple
  [1] ripe_banana
  ...

Example batch shapes:
  images: (32, 3, 224, 224)
  labels: (32,)

Train split:
  Total samples: 1234
  - ripe_apple        :    100 (  8.1%)
  - ripe_banana       :    120 (  9.7%)
  ...

Val split:
  ...

Test split:
  ...
```

This helps you quickly validate:

- Class balance
- Dataset size
- That data loading and transforms are working as expected.

---

## Troubleshooting

- **`ModuleNotFoundError: fruit_ripeness_dl`**  
  Make sure you are:
  - In the project root, and
  - Running commands via Poetry, e.g.:

  ```bash
  poetry run data-summary
  ```

- **Zip file not found when running `model-split`**  
  Check that:

  ```text
  data/raw/fruit_ripeness.zip
  ```

  exists, or pass a custom zip path:

  ```bash
  poetry run model-split --zip-path path/to/your.zip
  ```

- **CUDA not being used**  
  Check:

  ```bash
  poetry run python -c "import torch; print(torch.cuda.is_available())"
  ```

  And make sure to pass `--device cuda` to `train` or set `device="cuda"` in `TrainingConfig`.

---

This README should be enough for a newcomer to **install**, **prepare data**, **train**, and **evaluate** the fruit ripeness classifier end to end.
