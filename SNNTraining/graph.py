import torch
import re
import networkx as nx
import matplotlib.pyplot as plt
import snntorch as snn
from snntorch.export_nir import export_to_nir
import nir
import math

class Graph:
    def __init__(self, num_steps, num_inputs, seed=42):

        torch.manual_seed(seed)
        self.sample_data = torch.randn(num_steps, num_inputs)
        self.net = None  # Placeholder for the network model
        self.nir_model = None
        self.edges = None
        self.final_edges = None
        self.final_nodes = None
        self.graph = None
        self.recurrent_edges = None

    def log(self, dut=None):
        temp = "\n----- GRAPH -----\n"
        if dut is not None:
            dut._log.info(temp)
        else:
            print(temp)
        temp = "Recurrent edges:", self.recurrent_edges
        if dut is not None:
            dut._log.info(temp)
        else:
            print(temp)
        temp = "Nodes:", self.final_nodes
        if dut is not None:
            dut._log.info(temp)
        else:
            print(temp)
        temp = "Edges:", self.final_edges
        if dut is not None:
            dut._log.info(temp)
        else:
            print(temp)

    def export_model(self, net):
        self.net = net
        self.net.eval()  # Ensure model uses correct batch shape logic

        # Create sample input with shape [num_steps, batch_size, input_dim]
        batch_size = 10
        self.sample_data = torch.randn(self.net.num_steps, batch_size, self.net.fc1.in_features)

        # Export with time_first=True to match forward logic
        self.nir_model = export_to_nir(self.net, self.sample_data)

        self._save_nir_model("nir_model.h5")

    def _save_nir_model(self, filename):
        if self.nir_model is not None:
            self.nir_model = self._clean_nir_dict(self.nir_model)
            nir.write(filename, self.nir_model)

    def _clean_nir_dict(self, d):
        if isinstance(d, dict):
            return {k: self._clean_nir_dict(v) for k, v in d.items() if v is not None}
        elif isinstance(d, list):
            return [self._clean_nir_dict(v) for v in d if v is not None]
        else:
            return d

    def extract_edges(self):
        if self.nir_model is None:
            raise ValueError("NIR model has not been set. Please call export_model first.")
        
        text = str(self.nir_model)
        edges_match = re.search(r"edges=\[(.*?)\]", text)
        edges_str = edges_match.group(1) if edges_match else ""
        self.edges = eval(f"[{edges_str}]")

    def plot_graph(self):
        # Draw the graph
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=3100,
                node_color="blue", font_size=10, font_weight="bold",
                arrowsize=25, font_color="white")
        plt.title("Computational Graph")
        plt.savefig("nir_graph.png")  # Save before showing
        plt.show()

    def process_graph(self):
        if self.edges is None:
            raise ValueError("Edges have not been extracted. Please call extract_edges first.")
        
        G = nx.DiGraph(self.edges)  # Create a directed graph with the given edges

        # Identify all fully connected (fc) nodes
        fc_nodes = [node for node in G.nodes() if node.startswith('fc')]

        # Process each fully connected node
        for fc in fc_nodes:
            predecessors = list(G.predecessors(fc))  # List of predecessor nodes
            successors = list(G.successors(fc))      # List of successor nodes

            # Connect all predecessors to all successors, bypassing the fc node
            for pred in predecessors:
                for succ in successors:
                    G.add_edge(pred, succ)

            G.remove_node(fc)  # Remove the fc node from the graph

        # Identify recurrent edges (edges where there is a back edge creating a cycle)
        recurrent_edges = [(u, v) for u, v in G.edges() if G.has_edge(v, u)]

        # Remove duplicates from recurrent edges (since they are bidirectional)
        for u, v in recurrent_edges:
            if (v, u) in recurrent_edges:
                recurrent_edges.remove((v, u))

        # Process each recurrent edge
        for u, v in recurrent_edges:
            if G.has_edge(u, v):
                G.remove_edge(u, v)  # Remove the edge from u to v
            if G.has_edge(v, u):
                G.remove_edge(v, u)  # Remove the edge from v to u

            # Determine which node to remove and which to keep (keep Lif remove rec)
            if G.has_node(v) and "rec" in v:
                x = v
                y = u
            else:
                x = u
                y = v

            G.remove_node(x)  # Remove the node x
            G.add_edge(y, y)  # Add a self-loop on node y

            # Relabel the node y to its base name
            mapping = {
                y: y.split('.')[0]
            }
            nx.relabel_nodes(G, mapping, copy=False)  # Relabel the nodes in place

        self.graph = G
        self.final_nodes = G.nodes()
        self.final_edges = G.edges()
        self.recurrent_edges = recurrent_edges

    def get_neuron_connectivity(self, model):
        """
        Builds a dictionary showing which neurons connect to which, with weights.
        """
        connectivity = {}  # key = source, value = dict of {target: weight}

        # Input to Hidden (fc1)
        fc1_w = model.fc1.weight.data
        for to_hid in range(fc1_w.shape[0]):
            for from_in in range(fc1_w.shape[1]):
                weight = fc1_w[to_hid, from_in].item()
                if math.isfinite(weight) and weight != 0:
                    src = f"in_{from_in}"
                    tgt = f"lif1_{to_hid}"
                    connectivity.setdefault(src, {})[tgt] = weight

        # Recurrent within Hidden (lif1.recurrent)
        rec_w = model.lif1.recurrent.weight.data
        for from_hid in range(rec_w.shape[1]):
            src = f"lif1_{from_hid}"
            for to_hid in range(rec_w.shape[0]):
                weight = rec_w[to_hid, from_hid].item()
                if math.isfinite(weight) and weight != 0:
                    tgt = f"lif1_{to_hid}"
                    connectivity.setdefault(src, {})[tgt] = 1  # binary marker

        # Hidden to Output (fc2)
        fc2_w = model.fc2.weight.data
        for to_out in range(fc2_w.shape[0]):
            for from_hid in range(fc2_w.shape[1]):
                weight = fc2_w[to_out, from_hid].item()
                if math.isfinite(weight) and weight != 0:
                    src = f"lif1_{from_hid}"
                    tgt = f"lif2_{to_out}"
                    connectivity.setdefault(src, {})[tgt] = weight

        self.neuron_connectivity = connectivity  # save for later
        return connectivity
    def build_excitatory_matrix(self, model):
        """
        Builds a matrix of excitatory connections from lif1 to lif2 neurons.
        Only positive weights are retained; others are set to zero.
        """
        with torch.no_grad():
            # Extract fc2 weights: shape [num_lif2, num_lif1]
            fc2_w = model.fc2.weight.data  # lif1 -> lif2

            # Transpose to get [num_lif1, num_lif2]
            excitatory_matrix = fc2_w.T.clone()

            # Zero out inhibitory or null weights
            excitatory_matrix[excitatory_matrix <= 0] = 0

        return excitatory_matrix
    def build_recurrent_excitatory_matrix(self, model):
        """
        Builds a 512x512 matrix of excitatory recurrent connections (lif1 → lif1).
        Only positive weights are retained; others are set to zero.
        """
        with torch.no_grad():
            recurrent_w = model.lif1.recurrent.weight.data.clone()
            recurrent_w[recurrent_w <= 0] = 0
        return recurrent_w