"""Synthetic binary navigation dataset for SNN training."""

import numpy as np
import torch
from torch.utils.data import Dataset


class NavDataset(Dataset):
    """
    Synthetic binary navigation dataset.

    Each sample contains a spike train of shape (seq_len, n_neuron) and a
    binary label (0 = left, 1 = right).

    Neurons are divided into four equal groups:
        0: cue_left, 1: cue_right, 2: decision, 3: noise

    Parameters
    ----------
    seq_len:         Total number of time steps.
    n_neuron:        Total number of input neurons (must be divisible by n_input_symbols).
    recall_duration: Number of time steps the decision cue is active.
    p_group:         Probability of label 0 (left).
    f0:              Poisson firing probability for active groups.
    n_cues:          Number of cue pulses to emit.
    t_cue:           Upper bound for random cue onset times (exclusive).
    t_interval:      Time step at which the decision (recall) phase begins.
    n_input_symbols: Number of neuron groups (default 4).
    length:          Number of samples to generate.
    """

    def __init__(
        self,
        seq_len: int,
        n_neuron: int,
        recall_duration: int,
        p_group: float,
        f0: float = 0.5,
        n_cues: int = 7,
        t_cue: int = 100,
        t_interval: int = 150,
        n_input_symbols: int = 4,
        length: int = 100,
    ) -> None:
        super().__init__()
        data_list, label_list = [], []
        for _ in range(length):
            d, lbl = self._generate_sample(
                seq_len,
                n_neuron,
                recall_duration,
                p_group,
                f0,
                n_cues,
                t_cue,
                t_interval,
                n_input_symbols,
            )
            data_list.append(d)
            label_list.append(lbl)

        self.data = torch.tensor(np.stack(data_list), dtype=torch.float32)
        self.labels = torch.tensor(label_list, dtype=torch.long)

    @staticmethod
    def _generate_sample(
        seq_len: int,
        n_neuron: int,
        recall_duration: int,
        p_group: float,
        f0: float,
        n_cues: int,
        t_cue: int,
        t_interval: int,
        n_input_symbols: int,
    ) -> tuple[np.ndarray, int]:
        data = np.zeros((seq_len, n_neuron), dtype=np.float32)
        label = int(np.random.choice([0, 1], p=[p_group, 1.0 - p_group]))
        npg = n_neuron // n_input_symbols
        cue_group = 0 if label == 0 else 1

        # Cue pulses
        cue_times = np.random.randint(0, t_cue, size=n_cues)
        for t in cue_times:
            pattern = (np.random.rand(npg) < f0).astype(np.float32)
            data[t, cue_group * npg : (cue_group + 1) * npg] = pattern

        # Decision/recall phase
        for dt in range(recall_duration):
            t = t_interval + dt
            if t < seq_len:
                pattern = (np.random.rand(npg) < f0).astype(np.float32)
                data[t, 2 * npg : 3 * npg] = pattern

        # Background noise (noise group)
        for t in range(seq_len):
            pattern = (np.random.rand(npg) < f0).astype(np.float32)
            data[t, 3 * npg : 4 * npg] = pattern

        return data, label

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]
