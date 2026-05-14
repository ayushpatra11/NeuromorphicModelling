"""Tests for neuromorphic.config."""

import torch

from neuromorphic.config import HardwareSpecs, ModelConfig, Specs, Variables


class TestModelConfig:
    def test_default_values(self):
        cfg = ModelConfig()
        assert cfg.num_inputs == 40
        assert cfg.num_hidden == 512
        assert cfg.num_outputs == 2
        assert cfg.num_epochs == 100
        assert cfg.bs == 10
        assert 0 < cfg.p_group < 1

    def test_num_steps_derived(self):
        cfg = ModelConfig()
        expected = cfg.t_cue_spacing * cfg.n_cues + cfg.silence_duration + cfg.recall_duration
        assert cfg.num_steps == expected

    def test_device_is_torch_device(self):
        cfg = ModelConfig()
        assert isinstance(cfg.device, torch.device)

    def test_backward_compat_alias(self):
        v = Variables()
        assert isinstance(v, ModelConfig)


class TestHardwareSpecs:
    def test_direction_constants_unique(self):
        hw = HardwareSpecs()
        directions = [hw.EAST, hw.NORTH, hw.WEST, hw.SOUTH, hw.L1]
        assert len(set(directions)) == 5

    def test_masks_nonzero(self):
        hw = HardwareSpecs()
        for mask in [hw.E_MASK, hw.N_MASK, hw.W_MASK, hw.S_MASK]:
            assert mask != 0

    def test_backward_compat_alias(self):
        s = Specs()
        assert isinstance(s, HardwareSpecs)
