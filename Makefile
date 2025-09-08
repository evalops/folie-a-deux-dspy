PY=.venv/bin/python
VENV?=.venv
MODEL?=ollama_chat/llama3.1:8b
ALPHA?=0.0
ROUNDS?=6

.PHONY: setup install run run-alpha run-cli run-ablation run-verbose run-verbose-alpha sweep test test-verbose fmt clean help

help:
	@echo "Folie à Deux - Iterative LLM Agreement Training"
	@echo ""
	@echo "Available commands:"
	@echo "  setup         - Create virtual environment and install dependencies"
	@echo "  install       - Install package in development mode"
	@echo "  run           - Run experiment (legacy compatibility)"
	@echo "  run-alpha     - Run with alpha=0.1 (truth anchoring)"
	@echo "  run-cli       - Run using new CLI interface"
	@echo "  run-verbose   - Run with verbose debug logging"
	@echo "  run-verbose-alpha - Run verbose with truth anchoring (α=0.1)"
	@echo "  run-ablation  - Run ablation study with different alpha values"
	@echo "  sweep         - Run α parameter sweep for Pareto analysis"
	@echo "  test          - Run tests"
	@echo "  test-verbose  - Run tests with verbose output"
	@echo "  fmt           - Format code with ruff"
	@echo "  clean         - Clean build artifacts"
	@echo ""
	@echo "Environment variables:"
	@echo "  MODEL         - Model to use (default: $(MODEL))"
	@echo "  ALPHA         - Truth anchoring weight (default: $(ALPHA))"
	@echo "  ROUNDS        - Number of training rounds (default: $(ROUNDS))"

setup:
	python -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip
	. $(VENV)/bin/activate && pip install -r requirements.txt

install: setup
	. $(VENV)/bin/activate && pip install -e ".[dev]"

run:
        MODEL=$(MODEL) ALPHA=$(ALPHA) ROUNDS=$(ROUNDS) $(PY) scripts/folie_a_deux_ollama.py

run-alpha:
	$(MAKE) run ALPHA=0.1

run-cli:
	MODEL=$(MODEL) $(PY) -m folie_a_deux.main --alpha $(ALPHA) --rounds $(ROUNDS)

run-verbose:
	@echo "Starting verbose experiment with detailed logging..."
	@echo "Configuration: MODEL=$(MODEL), ALPHA=$(ALPHA), ROUNDS=$(ROUNDS)"
	@echo "Enabling DEBUG level logging for detailed output..."
	@echo ""
	MODEL=$(MODEL) $(PY) -m folie_a_deux.main --alpha $(ALPHA) --rounds $(ROUNDS) --log-level DEBUG

run-verbose-alpha:
	@echo "Starting verbose experiment with truth anchoring..."
	@echo "Configuration: MODEL=$(MODEL), ALPHA=0.1, ROUNDS=$(ROUNDS)"
	@echo "Enabling DEBUG level logging for detailed output..."
	@echo ""
	MODEL=$(MODEL) $(PY) -m folie_a_deux.main --alpha 0.1 --rounds $(ROUNDS) --log-level DEBUG

run-ablation:
	MODEL=$(MODEL) $(PY) -m folie_a_deux.main --ablation --rounds $(ROUNDS) --output results.json

sweep:
	@echo "Running α parameter sweep for Pareto analysis..."
	@for alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do \
		echo "Running with α=$$alpha"; \
                MODEL=$(MODEL) ALPHA=$$alpha ROUNDS=$(ROUNDS) $(PY) scripts/folie_a_deux_ollama.py; \
	done

test:
	$(PY) -m pytest tests/ -v

test-verbose:
	@echo "Running tests with maximum verbosity and coverage..."
	$(PY) -m pytest tests/ -v -s --tb=long

test-small:
        MODEL=$(MODEL) ALPHA=$(ALPHA) ROUNDS=2 $(PY) scripts/folie_a_deux_ollama.py

fmt:
	$(PY) -m ruff check --fix . || true
	$(PY) -m ruff format . || true

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
