import numpy as np
import numpy.random as rd
import torch
from torch.utils.data import Dataset, DataLoader
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt


class NavDataset(Dataset):
    """
    Custom dataset for Generating synthetic Binary Navigation data.

    This is similar to what was done in NeuroEval by 

    """
    def __init__(self, seq_len, n_neuron, recall_duration, p_group, f0=0.5,
                 n_cues=7, t_cue=100, t_interval=150, n_input_symbols=4, length=100):
        super(NavDataset, self).__init__()
        self.data = []
        self.labels = []

        for _ in range(length):
            data, label = self.generate_data(seq_len, n_neuron, recall_duration, p_group,
                                             f0, n_cues, t_cue, t_interval, n_input_symbols)
            self.data.append(data)
            self.labels.append(label)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def generate_data(self, seq_len, n_neuron, recall_duration, p_group,
                      f0, n_cues, t_cue, t_interval, n_input_symbols):
        """
        Generates one synthetic binary navigation sample (T x N),
        with probabilistic cues and Poisson noise.
        """
        data = np.zeros((seq_len, n_neuron), dtype=np.float32)
        label = np.random.choice([0, 1], p=[p_group, 1 - p_group])  # 0 = left, 1 = right

        # Group neuron allocation
        neurons_per_group = n_neuron // n_input_symbols
        cue_group = 0 if label == 0 else 1

        # Generate cue pulses (Poisson-based)
        cue_times = np.random.randint(0, t_cue, size=n_cues)
        for t in cue_times:
            spike_pattern = np.random.rand(neurons_per_group) < f0
            start_idx = cue_group * neurons_per_group
            data[t, start_idx:start_idx + neurons_per_group] = spike_pattern.astype(np.float32)

        # Decision pulse (3 steps during recall window)
        decision_time = t_interval
        for dt in range(recall_duration):
            t = decision_time + dt
            if t < seq_len:
                spike_pattern = np.random.rand(neurons_per_group) < f0
                start_idx = 2 * neurons_per_group  # decision group
                data[t, start_idx:start_idx + neurons_per_group] = spike_pattern.astype(np.float32)

        # Background noise (applied to noise group across time)
        noise_start = 3 * neurons_per_group
        for t in range(seq_len):
            spike_pattern = np.random.rand(neurons_per_group) < f0
            data[t, noise_start:noise_start + neurons_per_group] = spike_pattern.astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label
    

n_in = 20
t_cue_spacing = 15
silence_duration = 50
recall_duration = 20
seq_len = int(t_cue_spacing * 7 + silence_duration + recall_duration)
batch_size = 10
input_f0 = 40. / 100.
p_group = 0.3
n_cues = 7
t_cue = 10
t_interval = t_cue_spacing
n_input_symbols = 4

# Create dataset and dataloader
dataset = NavDataset(seq_len, n_in, recall_duration, p_group, input_f0, n_cues, t_cue, t_interval, n_input_symbols)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(len(dataset))

# Visualize the data
for spk_data, target_data in dataloader:
    data = spk_data[0]
    print(spk_data.size())

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_yticks(np.arange(0, 40, 2)) 
    splt.raster(data, ax, s=5, c="blue")

    plt.title("Input Sample")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()
    break  # Only display the first batch
