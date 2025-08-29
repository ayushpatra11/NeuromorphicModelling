####################################################################################
#
#   File Name: evaluate_spike_activity.py
#   Author:  Ayush Patra
#   Description: This script evaluates a trained SNN model on test data to extract
#                neuron-to-neuron connectivity based on observed spike activity over
#                a fixed number of time steps using the trained model.
#   Version History:
#       - 2025-07-30: Initial version
#
####################################################################################

import torch
import json
from collections import defaultdict
from model import SpikingNet
from dataset import NavDataset as BinaryNavigationDataset
from options import Variables
from torch.utils.data import DataLoader

def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 or set2 else 1.0

v = Variables()

# Load the trained model
model = SpikingNet(v)
model.load_state_dict(torch.load("best_trained.pth", map_location=torch.device("cpu")))
model.eval()

num_samples = 50

# Load the test dataset
test_set = BinaryNavigationDataset(seq_len=v.num_steps, n_neuron=v.num_inputs, recall_duration=v.recall_duration,
                                   p_group=v.p_group, f0=40./100., n_cues=v.n_cues, t_cue=v.t_cue,
                                   t_interval=v.t_cue_spacing, n_input_symbols=4, length=num_samples)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

print("Model and test data loaded successfully.")

num_neurons = v.num_hidden1

s=1

# Evaluate spike propagation
with torch.no_grad():
    source_to_all_targets_per_timestep = defaultdict(list)
    for data, _ in test_loader:
        # Initialize spike propagation tracker for this sample
        firing_matrix = torch.zeros(num_neurons, num_neurons)  # [source, target]
        neuron_spike_count = torch.zeros(num_neurons)

        _, _, spk1_rec = model(data, time_first=False)  # [time, batch, neuron]
        spk1_rec = spk1_rec.squeeze(1)  # [time, neuron]
        print(f"Sample{s}")
        for t in range(v.num_steps - 1):
            print(f"Evaluating timestep {t}...")
            sources = (spk1_rec[t] > 0).nonzero(as_tuple=True)[0]
            targets = (spk1_rec[t + 1] > 0).nonzero(as_tuple=True)[0]
            target_set = set(targets.tolist())
            for src in sources:
                src_id = src.item()
                neuron_spike_count[src_id] += 1
                source_to_all_targets_per_timestep[src_id].append((t, target_set))
                for tgt in targets:
                    firing_matrix[src_id][tgt.item()] += 1

        # Threshold to generate binary connectivity
        top_k = 100
        connectivity_matrix = torch.zeros_like(firing_matrix, dtype=torch.int)

        for src in range(num_neurons):
            if neuron_spike_count[src] == 0:
                continue
            topk_vals, topk_indices = torch.topk(firing_matrix[src], k=top_k)
            connectivity_matrix[src, topk_indices] = 1

        # Save matrix as JSON for this sample
        filename = f"../RoutingEval/data/connectivity_matrix/dynamic_connectivity_matrix_{s}.json"
        with open(filename, "w") as f:
            json.dump(connectivity_matrix.tolist(), f, indent=2)

        print(f"Connectivity matrix extracted and saved to {filename}.")
        print("Neuron firing counts over time steps:")
        for idx, count in enumerate(neuron_spike_count.tolist()):
            print(f"Neuron {idx}: {int(count)} spikes")
        s = s+1

print("\nChecking for path consistency (using Jaccard similarity threshold)...")

similarity_threshold = 0.98  # consider paths same if similarity >= 90%

def summarize_path_groups(src, path_groups, total_events):
    print(f"\nSummary for neuron {src}:")
    print(f"  - Total firings: {total_events}")
    print(f"  - Path groups identified: {len(path_groups)}")
    for i, group in enumerate(path_groups):
        example_targets = sorted(list(group['ref']))[:5]
        print(f"    Group {i+1}: {group['count']} times, {len(group['ref'])} targets, example: {example_targets}")

for src, all_target_sets in source_to_all_targets_per_timestep.items():
    path_groups = []
    for t, new_set in all_target_sets:
        matched = False
        for group in path_groups:
            if jaccard_similarity(group["ref"], new_set) >= similarity_threshold:
                group["timesteps"].append(t)
                group["count"] += 1
                matched = True
                break
        if not matched:
            path_groups.append({"ref": new_set, "timesteps": [t], "count": 1})

    summarize_path_groups(src, path_groups, len(all_target_sets))

    if len(path_groups) > 1:
        print(f"\nApproximate path variations for neuron {src}:")
        for i, group in enumerate(path_groups):
            print(f"  - Group {i+1} ({group['count']} times): {sorted(list(group['ref']))[:5]}... ({len(group['ref'])} targets) at timesteps {group['timesteps']}")
    else:
        print(f"Consistent path for neuron {src}")