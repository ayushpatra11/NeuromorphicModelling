"""Shared pytest fixtures for the neuromorphic test suite."""

import pytest
import torch

from neuromorphic.config import HardwareSpecs, ModelConfig
from neuromorphic.dataset import NavDataset
from neuromorphic.model import SpikingNet


@pytest.fixture(scope="session")
def cfg() -> ModelConfig:
    """Minimal ModelConfig for fast CPU-only testing."""
    c = ModelConfig()
    # Shrink the network and sequence so tests finish quickly
    c.num_hidden = 16
    c.num_epochs = 2
    c.bs = 4
    return c


@pytest.fixture(scope="session")
def hw() -> HardwareSpecs:
    return HardwareSpecs()


@pytest.fixture(scope="session")
def net(cfg: ModelConfig) -> SpikingNet:
    torch.manual_seed(0)
    return SpikingNet(cfg)


@pytest.fixture(scope="session")
def small_dataset(cfg: ModelConfig) -> NavDataset:
    return NavDataset(
        seq_len=cfg.num_steps,
        n_neuron=cfg.num_inputs,
        recall_duration=cfg.recall_duration,
        p_group=cfg.p_group,
        f0=0.4,
        n_cues=cfg.n_cues,
        t_cue=cfg.t_cue,
        t_interval=cfg.t_cue_spacing,
        n_input_symbols=4,
        length=10,
    )
