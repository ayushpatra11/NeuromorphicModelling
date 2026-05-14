# Comparative Analysis of Spike Routing Techniques in NoC-based Neuromorphic Computing

Head-to-head comparison of two multicast spike-routing schemes on a Neurogrid-inspired binary-tree Network-on-Chip:
**HBS (Hierarchical Bit-String)** vs. **Neurogrid-style subtree broadcast**.

---

## Headline results (lower waste is better)

| Neurons / Core | Cores | HBS Avg. Waste / sample | Broadcast Avg. / sample | Reduction |
|---:|---:|---:|---:|---:|
| 64 | 8  | **485.53**   | 727.41   | −33% |
| 32 | 16 | **1,051.01** | 1,699.39 | −38% |
| 16 | 32 | **1,923.70** | 3,860.48 | −50% |

---

## Project structure

```
NeuromorphicModelling/
├── neuromorphic/          # Core Python package
│   ├── config.py          # ModelConfig, HardwareSpecs
│   ├── model.py           # SpikingNet (3-layer SNN)
│   ├── dataset.py         # NavDataset (binary navigation task)
│   ├── trainer.py         # Training loop with pruning
│   ├── graph.py           # NIR export and connectivity extraction
│   ├── mapping.py         # Neuron-to-core mapping
│   ├── packets.py         # Spike packet recording and analysis
│   ├── sweep.py           # WandB sweep configuration
│   └── utils.py           # Hook utilities, message encoding
├── scripts/               # CLI entry points
│   ├── train.py           # Train the SNN
│   ├── export_nir.py      # Export to NIR + connectivity matrices
│   ├── evaluate_activity.py  # Record dynamic spike connectivity
│   ├── evaluate_results.py   # Plot HBS vs Neurogrid comparison
│   └── map_neurons.py     # Visualise neuron-to-core mapping
├── RoutingEval/
│   └── common/            # C++ routing simulator (Neurogrid + HBS)
├── tests/                 # pytest suite (no GPU required)
├── Makefile               # One-command workflow shortcuts
└── pyproject.toml
```

---

## Prerequisites

- Python 3.10 or later
- g++ with C++17 support (for the routing simulator)
- make

**Python dependencies** are declared in `pyproject.toml` and installed automatically in the next step.

---

## Installation

```bash
git clone https://github.com/ayushpatra11/NeuromorphicModelling.git
cd NeuromorphicModelling
pip install -e ".[dev]"
```

For result plotting (seaborn / pandas):
```bash
pip install -e ".[analysis]"
```

For WandB hyperparameter sweeps:
```bash
pip install -e ".[sweep]"
```

---

## Full pipeline

The pipeline runs in five stages. Each stage produces outputs consumed by the next.

### 1 — Train the SNN

Trains a 3-layer spiking neural network on a binary navigation task and saves the model weights.

```bash
python scripts/train.py --save model.pth
```

| Flag | Default | Description |
|---|---|---|
| `--save` | `model.pth` | Path to write the model checkpoint |
| `--sweep` | off | Launch a WandB hyperparameter sweep instead of a single run |

Training prints validation accuracy, precision, recall, and F1 at the end. The saved checkpoint is used by every subsequent stage.

**Makefile shortcut:**
```bash
make train                      # saves to model.pth
make train CHECKPOINT=best.pth  # saves to a custom path
```

---

### 2 — Export to NIR and extract static connectivity

Loads the trained model, exports it to NIR (Neuromorphic Intermediate Representation), and writes three connectivity JSON files plus a heatmap.

```bash
python scripts/export_nir.py --checkpoint model.pth --out-dir .
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `model.pth` | Path to the saved model |
| `--out-dir` | `.` | Directory for output files |

**Output files:**

| File | Description |
|---|---|
| `nir_model.h5` | Exported NIR model |
| `neuron_connectivity.json` | Full neuron-to-neuron connectivity |
| `excitatory_matrix.json` | Feed-forward excitatory weight matrix |
| `recurrent_excitatory_matrix.json` | Recurrent (LIF1 → LIF1) excitatory matrix |
| `recurrent_heatmap.png` | Heatmap visualisation of recurrent connectivity |
| `nir_graph.png` | Graph visualisation of the NIR model |

**Makefile shortcut:**
```bash
make export-nir
```

---

### 3 — Record dynamic spike connectivity

Runs the trained model on a set of navigation samples and, for each sample, builds a binary connectivity matrix from the recorded spike activity. These matrices are the traffic inputs to the routing simulator.

```bash
python scripts/evaluate_activity.py \
    --checkpoint model.pth \
    --num-samples 50 \
    --top-k 100 \
    --out-dir RoutingEval/data/connectivity_matrix
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `model.pth` | Path to the saved model |
| `--num-samples` | `50` | Number of test samples to evaluate |
| `--top-k` | `100` | Top-k threshold for binarising the connectivity |
| `--out-dir` | `RoutingEval/data/connectivity_matrix` | Output directory |

**Output:** one `dynamic_connectivity_matrix_<n>.json` file per sample (a 512 × 512 binary matrix where entry `[i][j] = 1` means neuron `i` drove neuron `j`).

**Makefile shortcut:**
```bash
make evaluate-activity
```

---

### 4 — Build the C++ routing simulator

Compiles both the Neurogrid and HBS simulators with g++ (C++17).

```bash
make build-cpp
```

Or manually:
```bash
make -C RoutingEval/common
```

The binary is placed at `RoutingEval/build/RunSimulator`.

---

### 5 — Run the routing simulation

Reads the connectivity matrices from step 3, runs both routing schemes (Neurogrid and HBS) across all samples and neuron-per-core configurations (16, 32, 64), and writes per-sample waste JSON reports.

```bash
cd RoutingEval/build && ./RunSimulator
```

The simulator loops over:
- **10 random mappings** (placement seeds)
- **3 core densities**: 16, 32, and 64 neurons per core
- **50 samples** per configuration

Report files are written to:
```
scratch/data/reports/mapping<N>/reports_512_<neurons_per_core>/
  neurogrid/waste_metrics_sample<i>.json
  hbs/waste_metrics_sample<i>.json
```

Each JSON contains `total_illegal_deliveries`, `per_neuron_waste`, and `per_core_waste`.

**Makefile shortcut:**
```bash
make simulate   # builds if needed, then runs
```

---

### 6 — Plot results

Reads the simulator output and generates per-sample bar charts and summary comparison plots for each core density configuration.

```bash
python scripts/evaluate_results.py \
    --reports-dir RoutingEval/data/reports \
    --out-dir ResultsEvaluation/figures
```

| Flag | Default | Description |
|---|---|---|
| `--reports-dir` | `RoutingEval/data/reports` | Root directory containing simulator report JSON files |
| `--out-dir` | `ResultsEvaluation/figures` | Directory to write figures |

**Output:** `waste_per_sample_512_<N>.png` and `waste_summary_512_<N>.png` for each configuration.

**Makefile shortcut:**
```bash
make evaluate-results
```

---

## All Makefile targets

```bash
make pipeline          # run everything end-to-end (train → NIR → activity → simulate → plot)
make install           # pip install -e ".[dev]"
make test              # run pytest
make test-cov          # run pytest with coverage report
make lint              # ruff check
make fmt               # ruff format (in-place)
make build-cpp         # compile C++ simulator
make train             # train SNN → model.pth
make export-nir        # export NIR + connectivity matrices
make evaluate-activity # record dynamic spike connectivity
make evaluate-results  # generate comparison plots
make simulate          # build C++ and run routing simulation
make clean             # remove all build and cache artifacts
```

Override variables at call time:
```bash
make train CHECKPOINT=run1.pth
make evaluate-activity CHECKPOINT=run1.pth CONN_DIR=data/conn
make evaluate-results REPORTS_DIR=scratch/data/reports FIGURES_DIR=figs
```

---

## Running tests

```bash
make test
# or directly:
pytest tests/ -v
```

The test suite covers config defaults, dataset shapes, model forward pass and spike values, neuron-to-core mapping, and packet utilities. No GPU is required.

---

## How it works

- **Broadcast (Neurogrid-style):** when a neuron fires, its spike climbs the binary tree to the Lowest Common Ancestor (LCA) of all destination cores and then floods the entire LCA subtree. Every leaf under the LCA receives the spike, even non-target cores.
- **HBS:** a compact child-mask header lets the spike follow only the branches that lead to real targets, skipping empty siblings. Waste is roughly proportional to the fraction of non-target leaves under the LCA.

**Waste** is counted as the number of times a core receives a spike when none of its neurons were a target.

---

## Why the difference grows with core count

As neurons are spread across more cores (smaller neurons-per-core), destination sets become more fragmented across the tree. Broadcast must flood an increasingly large subtree; HBS still only opens required branches. The gap therefore widens as core count increases.

---

## Limitations and next steps

- Current metric is delivery count, not energy or latency. Hop-cost or energy-per-hop models can be attached to the per-sample JSON logs.
- Only random placement is evaluated. Placement-aware or connectivity-driven mapping could reduce waste for both schemes.
- Larger or mesh-based topologies are not yet modelled.

---

## Cite

**Ayush Patra (2025).** *Comparative Analysis of Spike Routing Techniques in Network-on-Chip-based Neuromorphic Computing Platforms.* MSc dissertation, The University of Manchester.

---

## Acknowledgements

Thanks to Dr. Davide Bertozzi for his supervision and guidance throughout this project.
