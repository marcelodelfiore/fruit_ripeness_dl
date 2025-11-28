from fruit_ripeness_dl.config import TrainingConfig
from fruit_ripeness_dl.training.train_classifier import run_experiment as _run_experiment


def main() -> None:
    cfg = TrainingConfig(
        experiment_name="exp_baseline",
        learning_rate=1e-3,
        num_epochs=10,
    )
    _run_experiment(cfg)


if __name__ == "__main__":
    main()
