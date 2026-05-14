"""
Export a trained model to NIR format and generate connectivity matrices.

Usage
-----
  python scripts/export_nir.py --checkpoint model.pth [--out-dir .]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuromorphic.config import ModelConfig
from neuromorphic.graph import Graph
from neuromorphic.model import SpikingNet

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained SNN to NIR + connectivity JSON")
    parser.add_argument("--checkpoint", default="model.pth", help="Path to model checkpoint")
    parser.add_argument("--out-dir", default=".", help="Output directory for generated files")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig()
    model = SpikingNet(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    for name, param in model.named_parameters():
        data = param.data
        logger.info(
            "%s: shape=%s  NaN=%s  Inf=%s  AllZero=%s",
            name,
            tuple(data.shape),
            torch.isnan(data).any().item(),
            torch.isinf(data).any().item(),
            torch.all(data == 0).item(),
        )

    graph = Graph(cfg.num_steps, cfg.num_inputs)

    # Connectivity matrices
    connectivity = graph.get_neuron_connectivity(model)
    exc_matrix = graph.build_excitatory_matrix(model)
    rec_matrix = graph.build_recurrent_excitatory_matrix(model)

    logger.info("Excitatory connections: %d", (exc_matrix > 0).sum().item())
    logger.info("Recurrent excitatory connections: %d", (rec_matrix > 0).sum().item())

    with open(out / "neuron_connectivity.json", "w") as f:
        json.dump(connectivity, f, indent=2)
    with open(out / "excitatory_matrix.json", "w") as f:
        json.dump(exc_matrix.tolist(), f, indent=2)
    with open(out / "recurrent_excitatory_matrix.json", "w") as f:
        json.dump(rec_matrix.tolist(), f, indent=2)

    # NIR export
    graph.export_model(model)
    graph.extract_edges()
    graph.process_graph()
    graph.plot_graph(save_path=str(out / "nir_graph.png"))
    graph.log()

    with open(out / "processed_edges.json", "w") as f:
        json.dump(list(graph.final_edges), f)

    # Recurrent heatmap
    mat_np = np.array(rec_matrix.tolist())
    plt.figure(figsize=(12, 10))
    sns.heatmap(mat_np, cmap="viridis", cbar=True)
    plt.title("Recurrent Excitatory Connectivity (LIF1 → LIF1)")
    plt.xlabel("Target Neuron Index")
    plt.ylabel("Source Neuron Index")
    plt.tight_layout()
    plt.savefig(str(out / "recurrent_heatmap.png"))
    plt.close()
    logger.info("All outputs written to %s", out)


if __name__ == "__main__":
    main()
