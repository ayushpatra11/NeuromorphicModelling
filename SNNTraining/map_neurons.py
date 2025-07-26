####################################################################################
#
#   File Name: map_neurons.py
#   Author:  Ayush Patra
#   Description: This file utilizes the mapping class to map neurons to cores
#                based on the neurogrid approach (topology is a binary tree)
#   Version History:        
#       - 2025-07-02: Initial version
# 
#
####################################################################################

import json
from mapping import Mapping

# Define dummy layer sizes (e.g., lif1 with 512 neurons)
mem_potential_sizes = {
    "lif1": 512
}

# Define the capacity of each core
core_capacity = 35  # You can adjust this based on your simulation needs

# Create the Mapping instance and generate the mapping
mapper = Mapping(mem_potential_sizes=mem_potential_sizes)
mapper.core_capacity = core_capacity
mapper.map_neurons()  # Required to initialize internal mappings

# Retrieve the mappings
core_allocation, NIR_to_cores, neuron_to_core = mapper.get_mappings()

# Save the mappings to JSON files
with open("core_allocation.json", "w") as f:
    json.dump(core_allocation, f, indent=2)

with open("nir_to_cores.json", "w") as f:
    json.dump(NIR_to_cores, f, indent=2)

with open("neuron_to_core.json", "w") as f:
    json.dump(neuron_to_core, f, indent=2)

# Print a summary
print(f"Total cores used: {len(set(neuron_to_core.values()))}")
print("Sample neuron-to-core mappings:")
for k in list(neuron_to_core.keys())[:10]:
    print(f"{k} -> Core {neuron_to_core[k]}")

# --- Visualization of hierarchical tree structure of core communication ---
import networkx as nx
import matplotlib.pyplot as plt

# Helper function for hierarchical layout
def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.successors(root))
    if len(children) != 0:
        dx = width / 2
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
    return pos

def visualize_tree(num_cores):
    # Create a binary tree graph
    G = nx.DiGraph()
    for i in range(num_cores):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < num_cores:
            G.add_edge(i, left)
        if right < num_cores:
            G.add_edge(i, right)

    # Draw the graph with hierarchical layout
    pos = hierarchy_pos(G, 0)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, arrows=False, node_size=800, node_color='lightblue', font_size=10)
    plt.title("Neurogrid Core Routing Tree")
    plt.tight_layout()
    plt.savefig("neurogrid_tree.png")
    plt.show()

# Visualize the tree for the number of used cores
visualize_tree(len(set(neuron_to_core.values())))