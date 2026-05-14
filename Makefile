# ============================================================
# NeuromorphicModelling – Root Makefile
# ============================================================
# Typical workflow:
#   make install       create .venv and install the package
#   make pipeline      run everything end-to-end (train → NIR → activity → simulate → plot)
#   make test          run the full test suite
#   make lint          run ruff linter/formatter check
#   make build-cpp     compile the C++ routing simulator
#   make train         train the SNN (saves model.pth)
#   make export-nir    export trained model to NIR + connectivity JSON
#   make evaluate-activity   extract spike connectivity matrices
#   make evaluate-results    generate HBS vs Neurogrid comparison plots
#   make simulate      run the C++ routing simulation
#   make clean         remove build artifacts

.PHONY: all pipeline install venv test test-cov lint fmt fmt-check build-cpp clean-cpp \
        train export-nir evaluate-activity evaluate-results simulate clean

# ---------------------------------------------------------------------------
# Configuration – override on the command line if needed
# ---------------------------------------------------------------------------
VENV        ?= .venv
PYTHON      ?= $(VENV)/bin/python3
PYTEST      ?= $(VENV)/bin/pytest
CHECKPOINT  ?= model.pth
REPORTS_DIR ?= RoutingEval/data/reports
FIGURES_DIR ?= ResultsEvaluation/figures
CONN_DIR    ?= RoutingEval/data/connectivity_matrix
NIR_DIR     ?= outputs/nir

# ---------------------------------------------------------------------------
# Virtual environment + install
# ---------------------------------------------------------------------------
$(VENV)/bin/python3:
	python3 -m venv --without-pip $(VENV)
	$(VENV)/bin/python3 -m ensurepip --upgrade

venv: $(VENV)/bin/python3

install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,analysis]"

# ---------------------------------------------------------------------------
# Full end-to-end pipeline
# ---------------------------------------------------------------------------
pipeline: train export-nir evaluate-activity simulate evaluate-results

# Quick smoke-test pipeline: 1 epoch, 2 samples, 1 mapping, 16 neurons/core only
pipeline-quick:
	$(PYTHON) scripts/train.py --save $(CHECKPOINT) --num-epochs 1
	mkdir -p $(NIR_DIR)
	$(PYTHON) scripts/export_nir.py --checkpoint $(CHECKPOINT) --out-dir $(NIR_DIR)
	mkdir -p $(CONN_DIR)
	$(PYTHON) scripts/evaluate_activity.py \
		--checkpoint $(CHECKPOINT) \
		--num-samples 2 \
		--out-dir $(CONN_DIR)
	$(MAKE) -C RoutingEval/common clean
	$(MAKE) -C RoutingEval/common CXXFLAGS="-std=c++17 -I./ -I./nlohmann -DNUM_SAMPLES=2 -DNUM_MAPPINGS=1"
	mkdir -p RoutingEval/data/core_tree RoutingEval/data/connectivity_matrix RoutingEval/data/reports
	cd RoutingEval/build && ./RunSimulator
	mkdir -p $(FIGURES_DIR)
	$(PYTHON) scripts/evaluate_results.py \
		--reports-dir $(REPORTS_DIR) \
		--out-dir $(FIGURES_DIR)

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
	$(VENV)/bin/ruff check neuromorphic/ tests/ scripts/

fmt:
	$(VENV)/bin/ruff format neuromorphic/ tests/ scripts/

fmt-check:
	$(VENV)/bin/ruff format --check neuromorphic/ tests/ scripts/

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
	mkdir -p $(NIR_DIR)
	$(PYTHON) scripts/export_nir.py --checkpoint $(CHECKPOINT) --out-dir $(NIR_DIR)

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
	mkdir -p RoutingEval/data/core_tree RoutingEval/data/connectivity_matrix RoutingEval/data/reports
	@echo "Running C++ routing simulation…"
	cd RoutingEval/build && ./RunSimulator

# ---------------------------------------------------------------------------
# House-keeping
# ---------------------------------------------------------------------------
clean: clean-cpp
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	rm -rf outputs/
