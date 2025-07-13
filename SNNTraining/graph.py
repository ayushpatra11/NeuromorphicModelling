import torch
import re
import networkx as nx
import matplotlib.pyplot as plt
import snntorch as snn
import nir

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
        self.nir_model = snn.export_to_nir(self.net, self.sample_data)
        self._save_nir_model("nir_model.txt")

    def _save_nir_model(self, filename):
        if self.nir_model is not None:
            nir.write(filename, self.nir_model)

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
        pos = nx.spring_layout(self.graph)  # Positions for all nodes
        nx.draw(self.graph, pos, with_labels=True, node_size=3100, node_color="blue", font_size=10, font_weight="bold", arrowsize=25, font_color="white")
        plt.title("Computational Graph")
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