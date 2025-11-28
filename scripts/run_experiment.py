from fruit_ripeness_dl.config import TrainingConfig
from fruit_ripeness_dl.training.train_classifier import run_experiment


if __name__ == "__main__":
    cfg = TrainingConfig(
        experiment_name="exp_baseline",
        learning_rate=1e-3,
        num_epochs=10,
    )
    run_experiment(cfg)

