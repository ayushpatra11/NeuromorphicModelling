####################################################################################
#
#   File Name: export_nir.py
#   Author:  Ayush Patra
#   Description: Used to export the neuromorphic intermediate representation of the
#                trained model and provide a connectivity matrix for the excitatory
#                neurons
#   Version History:        
#       - 2025-07-02: Initial version
#       - 2025-07-02: Added support for newer version of snnTorch and nir_export
#       - 2025-07-02: Added support to derive weight matrix
#       - 2025-07-02: Used threshold and positive values to get excitatory neurons
#                     for the connectivity matrix.
#
####################################################################################

import torch
from graph import Graph
from model import SpikingNet  
import json
from options import Variables
from snntorch import surrogate
from sweep_handler import SweepHandler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


v = Variables()
sweep_handler = SweepHandler()



# Initialize Graph
graph = Graph(v.num_steps, v.num_inputs)

config = {
    'method': 'random',
    'metric': sweep_handler.metric,
    'parameters': sweep_handler.parameters_dict
}


spike_grad = surrogate.atan()


spike_grad = surrogate.sigmoid()


spike_grad = surrogate.fast_sigmoid()

# Load trained model
model = SpikingNet(
    v, 
    spike_grad=spike_grad, 
    learn_alpha=config.get('learn_alpha', False),
    learn_beta=config.get('learn_beta', False),
    learn_threshold=config.get('learn_threshold', False)
)
model.load_state_dict(torch.load("best_trained.pth"))
for name, param in model.named_parameters():
    data = param.data
    print(f"{name}: shape={data.shape}, NaN={torch.isnan(data).any().item()}, Inf={torch.isinf(data).any().item()}, Zero={torch.all(data == 0).item()}")
model.eval()

# Extract neuron-to-neuron connectivity matrix
connectivity = graph.get_neuron_connectivity(model)
excitatory_matrix = graph.build_excitatory_matrix(model)

# Verify excitatory matrix
print(f"Excitatory matrix shape: {excitatory_matrix.shape}")
positive_values = excitatory_matrix[excitatory_matrix > 0]
print(f"Number of excitatory connections: {positive_values.numel()}")
if positive_values.numel() > 0:
    print(f"Min positive weight: {positive_values.min().item()}")
    print(f"Max positive weight: {positive_values.max().item()}")

# Save connectivity and matrix
with open("neuron_connectivity.json", "w") as f:
    json.dump(connectivity, f, indent=2)

with open("excitatory_matrix.json", "w") as f:
    json.dump(excitatory_matrix.tolist(), f, indent=2)

# Extract and verify recurrent excitatory matrix
recurrent_matrix = graph.build_recurrent_excitatory_matrix(model)
print(f"Recurrent excitatory matrix shape: {recurrent_matrix.shape}")
recurrent_positive = recurrent_matrix[recurrent_matrix > 0]
print(f"Number of recurrent excitatory connections: {recurrent_positive.numel()}")
if recurrent_positive.numel() > 0:
    print(f"Min recurrent positive weight: {recurrent_positive.min().item()}")
    print(f"Max recurrent positive weight: {recurrent_positive.max().item()}")

with open("recurrent_excitatory_matrix.json", "w") as f:
    json.dump(recurrent_matrix.tolist(), f, indent=2)

# Export to NIR
graph.export_model(model)

# Optional: additional processing
graph.extract_edges()
graph.process_graph()
graph.plot_graph()
graph.log()
plt.savefig("nir_graph.png")

graph.extract_edges()
graph.process_graph()

with open("processed_edges.json", "w") as f:
    json.dump(list(graph.final_edges), f)


# Plot heatmap of recurrent excitatory matrix

with open("recurrent_excitatory_matrix.json", "r") as f:
    recurrent_matrix = json.load(f)

matrix_np = np.array(recurrent_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(matrix_np, cmap="viridis", cbar=True)
plt.title("Recurrent Excitatory Connectivity Heatmap (LIF1 → LIF1)")
plt.xlabel("Target Neuron Index")
plt.ylabel("Source Neuron Index")
plt.tight_layout()
plt.savefig("recurrent_heatmap.png")
plt.close()