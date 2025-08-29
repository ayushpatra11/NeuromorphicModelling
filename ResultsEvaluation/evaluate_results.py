import os
import re
import json
import math
from glob import glob
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def natural_sample_id(path: str) -> int:
    """Extract a numeric sample id from a filename like '..._17.json'.
    If multiple numbers exist, return the last one. If none, fall back to
    alphabetical order using a hash for stability.
    """
    nums = re.findall(r"(\d+)", os.path.basename(path))
    if nums:
        return int(nums[-1])
    # Stable fallback: map name to a deterministic pseudo-index
    return abs(hash(os.path.basename(path))) % (10**6)


def load_totals_from_dir(dir_path: str) -> List[Tuple[int, int]]:
    """Return list of (sample_id, total_illegal_deliveries) from all JSONs in dir."""
    files = sorted(glob(os.path.join(dir_path, "*.json")), key=natural_sample_id)
    out: List[Tuple[int, int]] = []
    for fp in files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            total = int(data.get("total_illegal_deliveries", 0))
            out.append((natural_sample_id(fp), total))
        except Exception as e:
            print(f"Warning: skipping '{fp}': {e}")
            continue
    return out


def align_samples(hbs: List[Tuple[int, int]], ng: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align by sample id. Returns (sample_ids, hbs_values, neuro_values)."""
    m: Dict[int, Dict[str, int]] = {}
    for sid, val in hbs:
        m.setdefault(sid, {})['hbs'] = val
    for sid, val in ng:
        m.setdefault(sid, {})['ng'] = val
    # Keep only samples present in both, sorted by id
    common_ids = sorted([sid for sid, d in m.items() if 'hbs' in d and 'ng' in d])
    hbs_vals = np.array([m[sid]['hbs'] for sid in common_ids], dtype=np.int64)
    ng_vals  = np.array([m[sid]['ng']  for sid in common_ids], dtype=np.int64)
    return np.array(common_ids, dtype=np.int64), hbs_vals, ng_vals


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_per_sample(sample_ids: np.ndarray, hbs_vals: np.ndarray, ng_vals: np.ndarray, title: str, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    x = np.arange(len(sample_ids))
    width = 0.45
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(111)
    ax.bar(x - width/2, hbs_vals, width, label='HBS')
    ax.bar(x + width/2, ng_vals,  width, label='Neurogrid')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Waste (total illegal deliveries)')
    ax.set_title(title)
    # Show fewer x tick labels if many samples
    if len(sample_ids) > 20:
        step = max(1, len(sample_ids)//20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(sample_ids[::step])
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(sample_ids)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_summary(hbs_vals: np.ndarray, ng_vals: np.ndarray, title: str, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    total_hbs = int(hbs_vals.sum())
    total_ng  = int(ng_vals.sum())
    avg_hbs = float(hbs_vals.mean()) if len(hbs_vals) else 0.0
    avg_ng  = float(ng_vals.mean())  if len(ng_vals) else 0.0

    categories = ['Total', 'Avg per sample']
    x = np.arange(len(categories))
    width = 0.35

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    bars_hbs = ax.bar(x - width/2, [total_hbs, avg_hbs], width, label='HBS')
    bars_ng = ax.bar(x + width/2, [total_ng,  avg_ng],  width, label='Neurogrid')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Waste (total / average)')
    ax.set_title(title)
    ax.legend()

    # Annotate bars with values
    for bars in [bars_hbs, bars_ng]:
        for bar in bars:
            height = bar.get_height()
            if isinstance(height, float) and not height.is_integer():
                label = f"{height:.2f}"
            else:
                label = f"{int(height)}"
            ax.text(bar.get_x() + bar.get_width() / 2, height * 1.01, label, ha='center', va='bottom')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    base = "../RoutingEval/data/reports"
    configs = ["512_16", "512_32", "512_64"]
    out_dir = "./figures"

    for cfg in configs:
        hbs_dir = os.path.join(base, f"reports_{cfg}", "hbs")
        ng_dir  = os.path.join(base, f"reports_{cfg}", "neurogrid")

        hbs = load_totals_from_dir(hbs_dir)
        ng  = load_totals_from_dir(ng_dir)

        if not hbs or not ng:
            print(f"Skipping {cfg}: missing data (hbs={len(hbs)} files, neurogrid={len(ng)} files)")
            continue

        sample_ids, hbs_vals, ng_vals = align_samples(hbs, ng)
        if len(sample_ids) == 0:
            print(f"Skipping {cfg}: no common samples between HBS and Neurogrid.")
            continue

        # Per-sample plot
        per_sample_title = f"Waste per sample — {cfg} (HBS vs Neurogrid)"
        per_sample_out   = os.path.join(out_dir, f"waste_per_sample_{cfg}.png")
        plot_per_sample(sample_ids, hbs_vals, ng_vals, per_sample_title, per_sample_out)
        print(f"Saved: {per_sample_out}")

        # Summary plot (Total and Average)
        summary_title = f"Summary — {cfg} (Total and Avg waste)"
        summary_out   = os.path.join(out_dir, f"waste_summary_{cfg}.png")
        plot_summary(hbs_vals, ng_vals, summary_title, summary_out)
        print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()