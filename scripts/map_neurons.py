"""
Visualise and save the neuron-to-core mapping for a binary-tree neuromorphic topology.

Usage
-----
  python scripts/map_neurons.py [--neurons 512] [--capacity 35] [--out-dir .]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuromorphic.mapping import Mapping
from neuromorphic.utils import setup_logging

setup_logging("map_neurons")
logger = logging.getLogger(__name__)


def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0.0, xcenter=0.5, pos=None):
    """Compute a hierarchical (top-down tree) layout for a directed graph."""
    if pos is None:
        pos = {}
    pos[root] = (xcenter, vert_loc)
    children = list(G.successors(root))
    if children:
        dx = width / len(children)
        nextx = xcenter - width / 2 + dx / 2
        for child in children:
            pos = hierarchy_pos(
                G,
                child,
                width=dx,
                vert_gap=vert_gap,
                vert_loc=vert_loc - vert_gap,
                xcenter=nextx,
                pos=pos,
            )
            nextx += dx
    return pos


def visualize_tree(num_cores: int, save_path: str = "neurogrid_tree.png") -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    for i in range(num_cores):
        for child in (2 * i + 1, 2 * i + 2):
            if child < num_cores:
                G.add_edge(i, child)

    pos = hierarchy_pos(G, 0)
    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos, with_labels=True, arrows=False, node_size=800, node_color="lightblue", font_size=10
    )
    plt.title("Neurogrid Core Routing Tree")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info("Tree visualisation saved → %s", save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and visualise neuron-to-core mapping")
    parser.add_argument("--neurons", type=int, default=512, help="Number of hidden neurons (lif1)")
    parser.add_argument("--capacity", type=int, default=35, help="Max neurons per core")
    parser.add_argument("--out-dir", default=".", help="Output directory for JSON and PNG files")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mapper = Mapping(mem_potential_sizes={"lif1": args.neurons})
    mapper.set_core_capacity(args.capacity)
    mapper.map_neurons()

    ca, nir, n2c = mapper.get_mappings()
    num_cores = len(set(n2c.values()))
    logger.info("Total cores used: %d", num_cores)
    logger.info("Sample mappings: %s", {k: n2c[k] for k in list(n2c)[:5]})

    with open(out / "core_allocation.json", "w") as f:
        json.dump(ca, f, indent=2)
    with open(out / "nir_to_cores.json", "w") as f:
        json.dump(nir, f, indent=2)
    with open(out / "neuron_to_core.json", "w") as f:
        json.dump(n2c, f, indent=2)

    logger.info("Mappings saved to %s/", out)
    visualize_tree(num_cores, save_path=str(out / "neurogrid_tree.png"))


if __name__ == "__main__":
    main()
