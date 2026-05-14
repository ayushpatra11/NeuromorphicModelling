# ============================================================
# NeuromorphicModelling – Root Makefile
# ============================================================
# Typical workflow:
#   make install       install the Python package in editable mode
#   make test          run the full test suite
#   make lint          run ruff linter/formatter check
#   make build-cpp     compile the C++ routing simulator
#   make train         train the SNN (saves model.pth)
#   make export-nir    export trained model to NIR + connectivity JSON
#   make evaluate-activity   extract spike connectivity matrices
#   make evaluate-results    generate HBS vs Neurogrid comparison plots
#   make simulate      run the C++ routing simulation
#   make clean         remove build artifacts

.PHONY: all install test test-cov lint fmt build-cpp clean \
        train export-nir evaluate-activity evaluate-results simulate

# ---------------------------------------------------------------------------
# Configuration – override on the command line if needed
# ---------------------------------------------------------------------------
PYTHON      ?= python
PYTEST      ?= pytest
CHECKPOINT  ?= model.pth
REPORTS_DIR ?= RoutingEval/data/reports
FIGURES_DIR ?= ResultsEvaluation/figures
CONN_DIR    ?= RoutingEval/data/connectivity_matrix

# ---------------------------------------------------------------------------
# Python package
# ---------------------------------------------------------------------------
install:
	$(PYTHON) -m pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test:
	$(PYTEST) tests/ -v --tb=short

test-cov:
	$(PYTEST) tests/ -v --tb=short --cov=neuromorphic --cov-report=term-missing --cov-report=xml

# ---------------------------------------------------------------------------
# Linting / formatting
# ---------------------------------------------------------------------------
lint:
	ruff check neuromorphic/ tests/ scripts/

fmt:
	ruff format neuromorphic/ tests/ scripts/

fmt-check:
	ruff format --check neuromorphic/ tests/ scripts/

# ---------------------------------------------------------------------------
# C++ simulator
# ---------------------------------------------------------------------------
build-cpp:
	$(MAKE) -C RoutingEval/common

clean-cpp:
	$(MAKE) -C RoutingEval/common clean

# ---------------------------------------------------------------------------
# Python workflows
# ---------------------------------------------------------------------------
train:
	$(PYTHON) scripts/train.py --save $(CHECKPOINT)

export-nir: $(CHECKPOINT)
	$(PYTHON) scripts/export_nir.py --checkpoint $(CHECKPOINT)

evaluate-activity: $(CHECKPOINT)
	mkdir -p $(CONN_DIR)
	$(PYTHON) scripts/evaluate_activity.py \
		--checkpoint $(CHECKPOINT) \
		--out-dir $(CONN_DIR)

evaluate-results:
	mkdir -p $(FIGURES_DIR)
	$(PYTHON) scripts/evaluate_results.py \
		--reports-dir $(REPORTS_DIR) \
		--out-dir $(FIGURES_DIR)

simulate: build-cpp
	@echo "Running C++ routing simulation…"
	cd RoutingEval/build && ./RunSimulator

# ---------------------------------------------------------------------------
# House-keeping
# ---------------------------------------------------------------------------
clean: clean-cpp
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	rm -f nir_model.h5 nir_graph.png recurrent_heatmap.png
	rm -f neuron_connectivity.json excitatory_matrix.json \
	      recurrent_excitatory_matrix.json processed_edges.json
