"""NIR export and computational graph utilities."""

from __future__ import annotations

import logging
import re

import networkx as nx
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Graph:
    """
    Wraps a trained SpikingNet, exports it to NIR, and provides helpers to
    extract the processed computational graph and weight matrices.
    """

    def __init__(self, num_steps: int, num_inputs: int, seed: int = 42) -> None:
        torch.manual_seed(seed)
        self.num_steps = num_steps
        self.num_inputs = num_inputs
        self.net: nn.Module | None = None
        self.nir_model = None
        self.edges: list | None = None
        self.final_edges = None
        self.final_nodes = None
        self.graph: nx.DiGraph | None = None
        self.recurrent_edges: list | None = None
        self.neuron_connectivity: dict | None = None

    def log(self, dut=None) -> None:
        lines = [
            "\n----- GRAPH -----",
            f"Recurrent edges: {self.recurrent_edges}",
            f"Nodes: {list(self.final_nodes) if self.final_nodes else None}",
            f"Edges: {list(self.final_edges) if self.final_edges else None}",
        ]
        for line in lines:
            if dut is not None:
                dut._log.info(line)
            else:
                logger.info(line)

    def export_model(self, net: nn.Module) -> None:
        """Export the network to NIR and save as nir_model.h5."""
        import nir
        from snntorch.export_nir import export_to_nir

        self.net = net
        self.net.eval()
        batch_size = 10
        sample = torch.randn(self.net.num_steps, batch_size, self.net.fc1.in_features)
        self.nir_model = export_to_nir(self.net, sample)
        self.nir_model = self._clean_nir_dict(self.nir_model)
        nir.write("nir_model.h5", self.nir_model)

    @staticmethod
    def _clean_nir_dict(d):
        if isinstance(d, dict):
            return {k: Graph._clean_nir_dict(v) for k, v in d.items() if v is not None}
        if isinstance(d, list):
            return [Graph._clean_nir_dict(v) for v in d if v is not None]
        return d

    def extract_edges(self) -> None:
        if self.nir_model is None:
            raise ValueError("NIR model not set. Call export_model first.")
        text = str(self.nir_model)
        match = re.search(r"edges=\[(.*?)\]", text)
        edges_str = match.group(1) if match else ""
        self.edges = eval(f"[{edges_str}]")  # noqa: S307 – NIR lib returns this format

    def process_graph(self) -> None:
        """
        Build a cleaned DiGraph:
        - Remove fc nodes (bypass with pred→succ edges).
        - Convert bidirectional edges to self-loops on the lif node.
        """
        if self.edges is None:
            raise ValueError("Edges not extracted. Call extract_edges first.")

        G = nx.DiGraph(self.edges)

        # Bypass fully-connected nodes
        for fc in [n for n in G.nodes() if n.startswith("fc")]:
            for pred in list(G.predecessors(fc)):
                for succ in list(G.successors(fc)):
                    G.add_edge(pred, succ)
            G.remove_node(fc)

        # Collapse recurrent (bidirectional) edges into self-loops
        recurrent = [(u, v) for u, v in G.edges() if G.has_edge(v, u)]
        seen = set()
        deduplicated = []
        for u, v in recurrent:
            if (v, u) not in seen:
                deduplicated.append((u, v))
            seen.add((u, v))

        for u, v in deduplicated:
            G.remove_edges_from([(u, v), (v, u)])
            lif_node = v if "rec" in u else u
            rec_node = u if "rec" in u else v
            if G.has_node(rec_node):
                G.remove_node(rec_node)
            G.add_edge(lif_node, lif_node)
            base = lif_node.split(".")[0]
            nx.relabel_nodes(G, {lif_node: base}, copy=False)

        self.graph = G
        self.final_nodes = G.nodes()
        self.final_edges = G.edges()
        self.recurrent_edges = deduplicated

    def plot_graph(self, save_path: str = "nir_graph.png") -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=3100,
            node_color="blue",
            font_size=10,
            font_weight="bold",
            arrowsize=25,
            font_color="white",
        )
        plt.title("Computational Graph")
        plt.savefig(save_path)
        plt.close()
        logger.info("Graph saved to %s", save_path)

    def get_neuron_connectivity(self, model: nn.Module) -> dict:
        """Build source→target weight dictionary from all layers."""
        import math

        connectivity: dict[str, dict[str, float]] = {}

        def _add(src: str, tgt: str, w: float) -> None:
            if math.isfinite(w) and w != 0:
                connectivity.setdefault(src, {})[tgt] = w

        fc1_w = model.fc1.weight.data
        for to_h in range(fc1_w.shape[0]):
            for from_i in range(fc1_w.shape[1]):
                _add(f"in_{from_i}", f"lif1_{to_h}", fc1_w[to_h, from_i].item())

        rec_w = model.lif1.recurrent.weight.data
        for from_h in range(rec_w.shape[1]):
            for to_h in range(rec_w.shape[0]):
                w = rec_w[to_h, from_h].item()
                if math.isfinite(w) and w != 0:
                    connectivity.setdefault(f"lif1_{from_h}", {})[f"lif1_{to_h}"] = 1

        fc2_w = model.fc2.weight.data
        for to_o in range(fc2_w.shape[0]):
            for from_h in range(fc2_w.shape[1]):
                _add(f"lif1_{from_h}", f"lif2_{to_o}", fc2_w[to_o, from_h].item())

        self.neuron_connectivity = connectivity
        return connectivity

    def build_excitatory_matrix(self, model: nn.Module) -> torch.Tensor:
        """Return [num_lif1, num_lif2] matrix keeping only positive fc2 weights."""
        with torch.no_grad():
            mat = model.fc2.weight.data.T.clone()
            mat[mat <= 0] = 0
        return mat

    def build_recurrent_excitatory_matrix(self, model: nn.Module) -> torch.Tensor:
        """Return [num_lif1, num_lif1] keeping only positive recurrent weights."""
        with torch.no_grad():
            mat = model.lif1.recurrent.weight.data.clone()
            mat[mat <= 0] = 0
        return mat
