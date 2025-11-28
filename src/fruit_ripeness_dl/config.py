from dataclasses import dataclass


@dataclass
class TrainingConfig:
    data_dir: str = "data/processed/fruit_ripeness"
    runs_dir: str = "runs"
    image_size: int = 128
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    experiment_name: str = "exp_baseline"
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"
