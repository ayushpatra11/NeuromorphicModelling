"""Spiking neural network model and evaluation metrics."""

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import RSynaptic, surrogate

from neuromorphic.config import ModelConfig


class Metrics:
    """Confusion-matrix-based evaluation metrics for binary classification."""

    def __init__(self) -> None:
        self.TP: int = 0
        self.FP: int = 0
        self.TN: int = 0
        self.FN: int = 0

    def return_predicted(self, output: torch.Tensor) -> torch.Tensor:
        """Return the predicted class from spike-count totals."""
        _, predicted = output.sum(dim=0).max(dim=1)
        return predicted

    def perf_measure(self, y_actual: torch.Tensor, y_hat: torch.Tensor) -> None:
        """Accumulate confusion-matrix counts."""
        for actual, pred in zip(y_actual.tolist(), y_hat.tolist()):
            if actual == pred == 1:
                self.TP += 1
            elif pred == 1 and actual != pred:
                self.FP += 1
            elif actual == pred == 0:
                self.TN += 1
            elif pred == 0 and actual != pred:
                self.FN += 1

    def precision(self) -> float:
        return self.TP / (self.TP + self.FP + 1e-8)

    def recall(self) -> float:
        return self.TP / (self.TP + self.FN + 1e-8)

    def f1_score(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r + 1e-8)

    def get_scores(self) -> tuple[int, int, int, int]:
        return self.TP, self.TN, self.FP, self.FN

    def reset(self) -> None:
        self.TP = self.FP = self.TN = self.FN = 0


class SpikingNet(nn.Module):
    """
    Three-layer recurrent SNN:
      Input (fc1) → Hidden RSynaptic (lif1) → Output Leaky (lif2)

    Forward returns (output_spikes, membrane_traces, hidden_spikes),
    all shaped [time, batch, neurons].
    """

    def __init__(
        self,
        cfg: ModelConfig,
        spike_grad=None,
        learn_alpha: bool = True,
        learn_beta: bool = True,
        learn_threshold: bool = True,
    ) -> None:
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(cfg.num_inputs, cfg.num_hidden, bias=False)
        self.lif1 = RSynaptic(
            alpha=0.9,
            beta=0.9,
            spike_grad=spike_grad,
            learn_alpha=learn_alpha,
            learn_threshold=learn_threshold,
            linear_features=cfg.num_hidden,
            reset_mechanism="subtract",
            reset_delay=False,
            all_to_all=True,
        )
        self.lif1.recurrent.bias = None  # biological plausibility

        self.fc2 = nn.Linear(cfg.num_hidden, cfg.num_outputs, bias=False)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.num_steps: int = cfg.num_steps

    # ------------------------------------------------------------------
    # Single time-step forward (used by DynamicInference)
    # ------------------------------------------------------------------
    def forward_one_ts(
        self,
        x: torch.Tensor,
        spk1: torch.Tensor,
        syn1: torch.Tensor,
        mem1: torch.Tensor,
        mem2: torch.Tensor,
        cur_sub: list | None = None,
        cur_add: list | None = None,
        time_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not time_first:
            x = x.transpose(1, 0)

        curr_sub_rec, curr_add_rec = [], []
        curr_sub_fc, curr_add_fc = [], []

        for lst, rec_dst, fc_dst in [
            (cur_sub, curr_sub_rec, curr_sub_fc),
            (cur_add, curr_add_rec, curr_add_fc),
        ]:
            if lst is None:
                continue
            for element in lst:
                if element[2] > 99:
                    fc_dst.append(element)
                else:
                    rec_dst.append(element)

        cur1 = self.fc1(x)
        spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)

        for multiplier, s_idx, d_idx in ((e[0], e[1], e[2]) for e in curr_sub_rec):
            w = self.lif1.recurrent.weight.data[d_idx, s_idx].item()
            syn1[d_idx] = syn1[d_idx] - w * multiplier

        for multiplier, s_idx, d_idx in ((e[0], e[1], e[2]) for e in curr_add_rec):
            w = self.lif1.recurrent.weight.data[d_idx, s_idx].item()
            syn1[d_idx] = syn1[d_idx] + w * multiplier

        cur2 = self.fc2(spk1)

        for multiplier, s_idx, d_idx in ((e[0], e[1], e[2] - 100) for e in curr_sub_fc):
            w = self.fc2.weight.data[d_idx, s_idx].item()
            cur2[d_idx] = cur2[d_idx] - w * multiplier

        for multiplier, s_idx, d_idx in ((e[0], e[1], e[2] - 100) for e in curr_add_fc):
            w = self.fc2.weight.data[d_idx, s_idx].item()
            cur2[d_idx] = cur2[d_idx] + w * multiplier

        spk2, mem2 = self.lif2(cur2, mem2)
        return spk2, spk1, syn1, mem1, mem2

    # ------------------------------------------------------------------
    # Full sequence forward
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, time_first: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [time, batch, inputs] if time_first else [batch, time, inputs]
            time_first: layout of x

        Returns:
            (output_spikes, membrane_traces, hidden_spikes)
            all shaped [time, batch, neurons]
        """
        if not time_first:
            x = x.permute(1, 0, 2).contiguous()

        batch_size = x.shape[1]
        spk1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
        syn1 = torch.zeros_like(spk1)
        mem1 = torch.zeros_like(spk1)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=x.device)

        assert x.shape[0] == self.num_steps, (
            f"Expected time dimension {self.num_steps}, got {x.shape[0]}"
        )

        spk2_rec, mem2_rec, spk1_rec = [], [], []
        for step in range(self.num_steps):
            cur1 = self.fc1(x[step])
            spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            spk1_rec.append(spk1.clone())

        return (
            torch.stack(spk2_rec, dim=0),
            torch.stack(mem2_rec, dim=0),
            torch.stack(spk1_rec, dim=0),
        )
