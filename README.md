# Comparative analysis of Spike Routing Techniques in NoC-based Neuromorphic Computing

Short, repo-ready summary of what this report + code do, the main results, and how to reproduce them.

---

## TL;DR
- **What:** Head-to-head comparison of two multicast spike-routing schemes:
  - **HBS (Hierarchical Bit-String)** vs.
  - **Neurogrid-style subtree broadcast (up-then-down).**
- **How:** Identical traffic, mappings, and tree NoC across both; primary metric is **waste** (non-target spike deliveries at leaf cores).
- **Findings:** **HBS consistently reduces waste by ~33–50%** as core count increases.

---

## Headline results (lower is better)

| Neurons/Core | Cores | HBS Avg. Waste / sample | Broadcast Avg. / sample | **Reduction (HBS vs Broadcast)** |
|---:|---:|---:|---:|---:|
| 64 | 8  | **485.53** | 727.41 | **−33%** |
| 32 | 16 | **1,051.01** | 1,699.39 | **−38%** |
| 16 | 32 | **1,923.70** | 3,860.48 | **−50%** |

(Also tracked totals per setting; trends match the averages.)

---

## What the repo contains
- **Spike recording & connectivity extraction** from a trained task-performing SNN (functional, not structural, connectivity).
- **Mapping generator** for 16/32/64 neurons per core (multiple seeds to capture placement variability).
- **Two matched tree NoC simulators** implementing:
  - **Subtree broadcast:** climb to LCA of destinations, then flood the subtree.
  - **HBS:** compact child-mask forwarding to open only required branches.
- **Post-processing**: merges per-neuron/per-sample logs into summary tables and figures.

---

## How to reproduce (high level)
1. **Train & record** the SNN on the task; dump spike trains.
2. **Build functional connectivity** (near-coincidence rule → top-k binarization → per-source destination sets).
3. **Generate mappings** for {16, 32, 64} neurons/core with multiple seeds.
4. **Run simulations** for each mapping under both routing schemes (same traffic, same tree).
5. **Aggregate & plot**: compute waste per sample / per neuron, plus totals; render summary tables & figures.


---

## Why the difference?
- **Broadcast** must visit **every leaf** under the chosen subtree → many non-target receptions when destinations are spread.
- **HBS** forwards **only along selected child branches** via a short mask → avoids empty siblings, containing spillover.

---

## Limitations & next steps
- Current metric is **deliveries** (not energy/latency). You can attach hop-cost/energy models to the logs next.
- Explore **per-level masks** (bigger headers, potentially even lower waste) and **placement-aware** mappings or meshed fabrics.

---

## Cite
**Ayush Patra (2025).** *Comparative analysis of Spike Routing Techniques in Network-on-Chip-based Neuromorphic Computing Platforms.* MSc dissertation, The University of Manchester.

---

## Acknowledgements
Thanks to Dr. Davide Bertozzi for his supervision and guidance throughout this project.
