"""Tests for neuromorphic.model (SpikingNet and Metrics)."""

import pytest
import torch

from neuromorphic.model import Metrics, SpikingNet


class TestSpikingNet:
    def test_instantiation(self, cfg):
        net = SpikingNet(cfg)
        assert net.fc1.in_features == cfg.num_inputs
        assert net.fc1.out_features == cfg.num_hidden
        assert net.fc2.in_features == cfg.num_hidden
        assert net.fc2.out_features == cfg.num_outputs

    def test_no_bias(self, net):
        assert net.fc1.bias is None
        assert net.fc2.bias is None

    def test_forward_time_first_output_shape(self, cfg, net):
        batch, T = 2, cfg.num_steps
        x = torch.zeros(T, batch, cfg.num_inputs)
        spk2, mem2, spk1 = net(x, time_first=True)
        assert spk2.shape == (T, batch, cfg.num_outputs)
        assert mem2.shape == (T, batch, cfg.num_outputs)
        assert spk1.shape == (T, batch, cfg.num_hidden)

    def test_forward_batch_first_output_shape(self, cfg, net):
        batch, T = 3, cfg.num_steps
        x = torch.zeros(batch, T, cfg.num_inputs)
        spk2, mem2, spk1 = net(x, time_first=False)
        assert spk2.shape == (T, batch, cfg.num_outputs)

    def test_forward_spikes_binary(self, cfg, net):
        x = torch.randn(cfg.num_steps, 1, cfg.num_inputs)
        spk2, _, _ = net(x)
        unique = spk2.unique()
        assert all(v.item() in (0.0, 1.0) for v in unique)

    def test_wrong_time_dim_raises(self, cfg, net):
        x = torch.zeros(cfg.num_steps + 5, 1, cfg.num_inputs)
        with pytest.raises(AssertionError):
            net(x)

    def test_forward_one_ts_shape(self, cfg, net):
        x = torch.zeros(1, cfg.num_inputs)
        spk1_h = torch.zeros(1, cfg.num_hidden)
        syn1 = torch.zeros_like(spk1_h)
        mem1 = torch.zeros_like(spk1_h)
        mem2 = torch.zeros(1, cfg.num_outputs)
        spk2, spk1_out, syn1_out, mem1_out, mem2_out = net.forward_one_ts(
            x, spk1_h, syn1, mem1, mem2
        )
        assert spk2.shape == (1, cfg.num_outputs)
        assert spk1_out.shape == (1, cfg.num_hidden)


class TestMetrics:
    def test_initial_zeros(self):
        m = Metrics()
        assert m.TP == m.FP == m.TN == m.FN == 0

    def test_perf_measure_perfect_binary(self):
        m = Metrics()
        y_actual = torch.tensor([0, 1, 0, 1])
        y_hat = torch.tensor([0, 1, 0, 1])
        m.perf_measure(y_actual, y_hat)
        assert m.TP == 2
        assert m.TN == 2
        assert m.FP == 0
        assert m.FN == 0

    def test_precision_recall_f1_range(self):
        m = Metrics()
        m.TP, m.FP, m.TN, m.FN = 8, 2, 5, 1
        assert 0 <= m.precision() <= 1
        assert 0 <= m.recall() <= 1
        assert 0 <= m.f1_score() <= 1

    def test_f1_harmonic_mean(self):
        m = Metrics()
        m.TP, m.FP, m.TN, m.FN = 4, 0, 4, 0
        assert abs(m.precision() - 1.0) < 1e-3
        assert abs(m.recall() - 1.0) < 1e-3
        assert abs(m.f1_score() - 1.0) < 1e-3

    def test_reset(self):
        m = Metrics()
        m.TP, m.FP, m.TN, m.FN = 5, 3, 2, 1
        m.reset()
        assert m.TP == m.FP == m.TN == m.FN == 0

    def test_return_predicted_shape(self, net, cfg):
        batch = 4
        x = torch.zeros(cfg.num_steps, batch, cfg.num_inputs)
        spk2, _, _ = net(x)
        m = Metrics()
        preds = m.return_predicted(spk2)
        assert preds.shape == (batch,)
