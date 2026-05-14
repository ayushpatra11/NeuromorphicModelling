"""Tests for neuromorphic.dataset."""

import torch

from neuromorphic.dataset import NavDataset


class TestNavDataset:
    def test_length(self, small_dataset):
        assert len(small_dataset) == 10

    def test_item_shapes(self, cfg, small_dataset):
        data, label = small_dataset[0]
        assert data.shape == (cfg.num_steps, cfg.num_inputs)
        assert label.shape == ()

    def test_data_dtype(self, small_dataset):
        data, label = small_dataset[0]
        assert data.dtype == torch.float32
        assert label.dtype == torch.long

    def test_labels_binary(self, small_dataset):
        for _, lbl in small_dataset:
            assert lbl.item() in (0, 1)

    def test_data_values_in_range(self, small_dataset):
        for data, _ in small_dataset:
            assert data.min().item() >= 0.0
            assert data.max().item() <= 1.0

    def test_larger_dataset(self, cfg):
        ds = NavDataset(
            seq_len=cfg.num_steps,
            n_neuron=cfg.num_inputs,
            recall_duration=cfg.recall_duration,
            p_group=0.5,
            f0=0.4,
            n_cues=cfg.n_cues,
            t_cue=cfg.t_cue,
            t_interval=cfg.t_cue_spacing,
            n_input_symbols=4,
            length=50,
        )
        assert len(ds) == 50
        labels = torch.stack([ds[i][1] for i in range(50)])
        # Both classes should appear in 50 samples (extremely unlikely to be all one class)
        assert labels.sum().item() > 0
        assert labels.sum().item() < 50

    def test_determinism_given_seed(self, cfg):
        """Same numpy seed produces same dataset."""
        import numpy as np

        np.random.seed(42)
        ds1 = NavDataset(
            seq_len=cfg.num_steps,
            n_neuron=cfg.num_inputs,
            recall_duration=cfg.recall_duration,
            p_group=0.5,
            f0=0.4,
            n_cues=cfg.n_cues,
            t_cue=cfg.t_cue,
            t_interval=cfg.t_cue_spacing,
            n_input_symbols=4,
            length=5,
        )
        np.random.seed(42)
        ds2 = NavDataset(
            seq_len=cfg.num_steps,
            n_neuron=cfg.num_inputs,
            recall_duration=cfg.recall_duration,
            p_group=0.5,
            f0=0.4,
            n_cues=cfg.n_cues,
            t_cue=cfg.t_cue,
            t_interval=cfg.t_cue_spacing,
            n_input_symbols=4,
            length=5,
        )
        assert torch.allclose(ds1.data, ds2.data)
