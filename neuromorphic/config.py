"""Configuration classes for model hyperparameters and hardware specifications."""

import os

import torch


class ModelConfig:
    """Model and training hyperparameters for the SNN binary-navigation task."""

    def __init__(self) -> None:
        self.num_inputs: int = 40
        self.num_hidden: int = 512
        self.num_outputs: int = 2
        self.core_capacity: int = 25
        self.num_epochs: int = 100
        self.lr: float = 1e-4
        self.target_fr: float = 1.0
        self.bs: int = 10
        self.num_cores: int = 8
        self.target_sparsity: float = 1.0
        self.wandb_key: str | None = os.environ.get("WANDB_API_KEY")

        self.train: bool = False

        self.recall_duration: int = 20
        self.t_cue_spacing: int = 15
        self.silence_duration: int = 30
        self.n_cues: int = 7
        self.t_cue: int = 10
        self.p_group: float = 0.3

    @property
    def num_steps(self) -> int:
        return int(self.t_cue_spacing * self.n_cues + self.silence_duration + self.recall_duration)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HardwareSpecs:
    """Hardware specifications for the neuromorphic packet routing layer."""

    def __init__(self) -> None:
        self.ADDR_W: int = 5
        self.MSG_W: int = 10
        self.NUM_PACKETS_P_INJ: int = 20

        self.EAST: int = 0
        self.NORTH: int = 1
        self.WEST: int = 2
        self.SOUTH: int = 3
        self.L1: int = 4

        self.SID: int = 0b00001
        self.E_MASK: int = 0b10000
        self.N_MASK: int = 0b01000
        self.W_MASK: int = 0b00100
        self.S_MASK: int = 0b00010


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (old names used in legacy scripts)
# ---------------------------------------------------------------------------
Variables = ModelConfig
Specs = HardwareSpecs
