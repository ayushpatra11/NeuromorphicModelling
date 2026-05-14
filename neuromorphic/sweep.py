"""WandB hyperparameter sweep configuration."""


class SweepHandler:
    """Defines the sweep metric and parameter grid for WandB random search."""

    def __init__(self) -> None:
        self.metric = {"name": "loss", "goal": "minimize"}

        self.parameters_dict = {
            "learning_rate": {"values": [0.001, 0.0001, 0.00001]},
            "optimizer": {"values": ["Adam", "AdamW"]},
            "learn_alpha": {"values": [True, False]},
            "learn_beta": {"values": [True, False]},
            "learn_threshold": {"values": [True, False]},
            "surrogate_gradient": {"values": ["atan", "sigmoid", "fast_sigmoid"]},
            "batch_size": {"values": [10]},
        }
