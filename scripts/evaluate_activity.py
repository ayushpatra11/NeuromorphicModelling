"""
Evaluate spike-propagation connectivity from a trained model.

For each test sample, records spike activity over all time steps, builds a
[source × target] firing matrix, and saves a binary connectivity matrix as JSON.

Usage
-----
  python scripts/evaluate_activity.py \\
    --checkpoint model.pth \\
    --num-samples 50 \\
    --top-k 100 \\
    --out-dir RoutingEval/data/connectivity_matrix
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuromorphic.config import ModelConfig
from neuromorphic.dataset import NavDataset
from neuromorphic.model import SpikingNet
from neuromorphic.utils import setup_logging

setup_logging("evaluate_activity")
logger = logging.getLogger(__name__)


def jaccard_similarity(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def evaluate_activity(
    model: SpikingNet,
    dataset: NavDataset,
    cfg: ModelConfig,
    top_k: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    source_to_targets: dict[int, list[tuple[int, set]]] = defaultdict(list)

    with torch.no_grad():
        for sample_idx, (data, _) in enumerate(loader, start=1):
            firing_matrix = torch.zeros(cfg.num_hidden, cfg.num_hidden)
            spike_count = torch.zeros(cfg.num_hidden)

            _, _, spk1_rec = model(data, time_first=False)
            spk1_rec = spk1_rec.squeeze(1)  # [time, neurons]

            logger.info("Processing sample %d", sample_idx)
            for t in range(cfg.num_steps - 1):
                sources = (spk1_rec[t] > 0).nonzero(as_tuple=True)[0]
                targets = (spk1_rec[t + 1] > 0).nonzero(as_tuple=True)[0]
                tgt_set = set(targets.tolist())
                for src in sources:
                    sid = src.item()
                    spike_count[sid] += 1
                    source_to_targets[sid].append((t, tgt_set))
                    for tgt in targets:
                        firing_matrix[sid][tgt.item()] += 1

            # Threshold to binary connectivity
            conn = torch.zeros_like(firing_matrix, dtype=torch.int)
            for src in range(cfg.num_hidden):
                if spike_count[src] == 0:
                    continue
                k = min(top_k, firing_matrix.shape[1])
                _, topk_idx = torch.topk(firing_matrix[src], k=k)
                conn[src, topk_idx] = 1

            fname = out_dir / f"dynamic_connectivity_matrix_{sample_idx}.json"
            with open(fname, "w") as f:
                json.dump(conn.tolist(), f, indent=2)
            logger.info("Saved connectivity matrix → %s", fname)

    # Jaccard path-consistency report
    threshold = 0.98
    logger.info("Checking path consistency (Jaccard threshold=%.2f)…", threshold)
    for src, events in source_to_targets.items():
        groups: list[dict] = []
        for t, tgt_set in events:
            matched = False
            for grp in groups:
                if jaccard_similarity(grp["ref"], tgt_set) >= threshold:
                    grp["timesteps"].append(t)
                    grp["count"] += 1
                    matched = True
                    break
            if not matched:
                groups.append({"ref": tgt_set, "timesteps": [t], "count": 1})

        if len(groups) > 1:
            logger.info(
                "Neuron %d: %d path variations (firings=%d)",
                src,
                len(groups),
                len(events),
            )
        else:
            logger.info("Neuron %d: consistent path (firings=%d)", src, len(events))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract dynamic spike connectivity matrices")
    parser.add_argument("--checkpoint", default="model.pth")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--out-dir", default="RoutingEval/data/connectivity_matrix")
    args = parser.parse_args()

    cfg = ModelConfig()
    model = SpikingNet(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dataset = NavDataset(
        seq_len=cfg.num_steps,
        n_neuron=cfg.num_inputs,
        recall_duration=cfg.recall_duration,
        p_group=cfg.p_group,
        f0=40.0 / 100.0,
        n_cues=cfg.n_cues,
        t_cue=cfg.t_cue,
        t_interval=cfg.t_cue_spacing,
        n_input_symbols=4,
        length=args.num_samples,
    )

    evaluate_activity(model, dataset, cfg, args.top_k, Path(args.out_dir))


if __name__ == "__main__":
    main()
