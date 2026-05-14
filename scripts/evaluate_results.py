"""
Generate HBS vs Neurogrid routing-waste comparison plots.

Usage
-----
  python scripts/evaluate_results.py \\
    --reports-dir RoutingEval/data/reports \\
    --out-dir ResultsEvaluation/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _natural_sample_id(path: str) -> int:
    nums = re.findall(r"(\d+)", os.path.basename(path))
    return int(nums[-1]) if nums else abs(hash(os.path.basename(path))) % (10**6)


def load_totals(dir_path: str) -> list[tuple[int, int]]:
    files = sorted(glob(os.path.join(dir_path, "*.json")), key=_natural_sample_id)
    out: list[tuple[int, int]] = []
    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
            out.append((_natural_sample_id(fp), int(data.get("total_illegal_deliveries", 0))))
        except Exception as exc:
            logger.warning("Skipping '%s': %s", fp, exc)
    return out


def align_samples(
    hbs: list[tuple[int, int]], ng: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m: dict[int, dict[str, int]] = {}
    for sid, val in hbs:
        m.setdefault(sid, {})["hbs"] = val
    for sid, val in ng:
        m.setdefault(sid, {})["ng"] = val
    common = sorted(sid for sid, d in m.items() if "hbs" in d and "ng" in d)
    return (
        np.array(common, dtype=np.int64),
        np.array([m[s]["hbs"] for s in common], dtype=np.int64),
        np.array([m[s]["ng"] for s in common], dtype=np.int64),
    )


def plot_per_sample(
    sample_ids: np.ndarray, hbs: np.ndarray, ng: np.ndarray, title: str, out_path: str
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    x, w = np.arange(len(sample_ids)), 0.45
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w / 2, hbs, w, label="HBS")
    ax.bar(x + w / 2, ng, w, label="Neurogrid")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Waste (total illegal deliveries)")
    ax.set_title(title)
    step = max(1, len(sample_ids) // 20) if len(sample_ids) > 20 else 1
    ax.set_xticks(x[::step])
    ax.set_xticklabels(sample_ids[::step])
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_summary(hbs: np.ndarray, ng: np.ndarray, title: str, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    total_h, total_n = int(hbs.sum()), int(ng.sum())
    avg_h = float(hbs.mean()) if len(hbs) else 0.0
    avg_n = float(ng.mean()) if len(ng) else 0.0
    cats = ["Total", "Avg per sample"]
    x, w = np.arange(2), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_h = ax.bar(x - w / 2, [total_h, avg_h], w, label="HBS")
    bars_n = ax.bar(x + w / 2, [total_n, avg_n], w, label="Neurogrid")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Waste")
    ax.set_title(title)
    ax.legend()
    for bars in [bars_h, bars_n]:
        for bar in bars:
            h = bar.get_height()
            label = f"{h:.2f}" if isinstance(h, float) and not h.is_integer() else f"{int(h)}"
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01, label, ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HBS vs Neurogrid routing waste")
    parser.add_argument("--reports-dir", default="RoutingEval/data/reports")
    parser.add_argument("--out-dir", default="ResultsEvaluation/figures")
    args = parser.parse_args()

    configs = ["512_16", "512_32", "512_64"]
    for cfg in configs:
        hbs_dir = os.path.join(args.reports_dir, f"reports_{cfg}", "hbs")
        ng_dir = os.path.join(args.reports_dir, f"reports_{cfg}", "neurogrid")

        hbs = load_totals(hbs_dir)
        ng = load_totals(ng_dir)

        if not hbs or not ng:
            logger.warning("Skipping %s: missing data (hbs=%d, ng=%d)", cfg, len(hbs), len(ng))
            continue

        ids, hbs_vals, ng_vals = align_samples(hbs, ng)
        if len(ids) == 0:
            logger.warning("Skipping %s: no common samples.", cfg)
            continue

        per_path = os.path.join(args.out_dir, f"waste_per_sample_{cfg}.png")
        plot_per_sample(ids, hbs_vals, ng_vals, f"Waste per sample — {cfg}", per_path)
        logger.info("Saved: %s", per_path)

        sum_path = os.path.join(args.out_dir, f"waste_summary_{cfg}.png")
        plot_summary(hbs_vals, ng_vals, f"Summary — {cfg}", sum_path)
        logger.info("Saved: %s", sum_path)


if __name__ == "__main__":
    main()
