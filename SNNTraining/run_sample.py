import torch
from model import SpikingNet
from options import Variables
from snntorch import surrogate

# Step 1: Setup config and load model
v = Variables()
spike_grad = surrogate.fast_sigmoid()

model = SpikingNet(
    v,
    spike_grad=spike_grad,
    learn_alpha=False,
    learn_beta=False,
    learn_threshold=False
)
model.load_state_dict(torch.load("best_snn.pth"))
model.eval()

# Step 2: Create dummy input with correct shape
# Shape: [num_steps, batch_size, num_inputs]
x = torch.randn(v.num_steps, 10, v.num_inputs)

# Step 3: Forward pass
with torch.no_grad():
    output_spikes, mem_traces = model(x, time_first=True)

# Step 4: Analyze output
# Print shape and a sample timestep output
print("Output shape:", output_spikes.shape)  # [num_steps, batch_size, num_outputs]
print("Membrane traces shape:", mem_traces.shape)

# Optional: Compute predicted label from output spikes
predicted = output_spikes.sum(dim=0).argmax(dim=1)
#print("Predicted class:", predicted.item())