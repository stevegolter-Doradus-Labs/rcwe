# RCWE developer shortcuts. Everything here is a thin wrapper around the `rcwe` command,
# so the same steps work without make (see `rcwe --help`).
#
# The interpreter is the project's virtualenv if there is one, otherwise whatever
# `python3` is on your PATH. Override with:  make test PYTHON=/path/to/python

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else command -v python3 || command -v python; fi)
RCWE := $(PYTHON) -m rcwe

.DEFAULT_GOAL := help
.PHONY: help install quickstart doctor serve ui test fast bench extreme dashboard realworld phase2 \
        train-corpus custom-data v09 predict benchmarks clean

help:  ## Show this help
	@echo "RCWE make targets (python: $(PYTHON))"
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install RCWE (editable) with test dependencies into the current environment
	$(PYTHON) -m pip install -e ".[dev]"

quickstart:  ## 20-second guided tour on built-in demo data
	@$(RCWE) quickstart

doctor:  ## Check that this machine can run RCWE
	@$(RCWE) doctor

serve:  ## RCWE Studio in the browser: upload a log, explore the report
	@$(RCWE) serve

ui:  ## Mission Control: live dashboard of the engine on the built-in sandbox
	@$(RCWE) ui

test:  ## Full automated test suite
	@$(PYTHON) -m compileall -q rcwe tests
	@$(PYTHON) -m pytest -q

fast:  ## Quick smoke tests (core maths + toolkit data layer)
	@$(PYTHON) -m compileall -q rcwe tests
	@$(PYTHON) -m pytest -q tests/test_invariance.py tests/test_plasticity_and_mdo.py tests/test_toolkit_table.py

# ---- research benchmark suites (reports go to ./rcwe_output) ---------------------------
bench:  ## Multi-seed validation benchmark
	@$(RCWE) bench validate

extreme:  ## Context-length scaling sweep + symbolic conservation task
	@$(RCWE) bench extreme

dashboard:  ## CSV/JSON ingestion, code-trace task and the static v0.5 dashboard
	@$(RCWE) bench dashboard

realworld:  ## ETT / SWaT / NASA-style regime-shift suite
	@$(RCWE) bench realworld

phase2:  ## Metal kernel scaling, self-healing controller, RCWE-Deep language model
	@$(RCWE) bench phase2

train-corpus:  ## Train RCWE-Deep on the repository's own source code
	@$(RCWE) bench corpus

custom-data:  ## Scan datasets/custom/ with the legacy predictor
	@$(RCWE) bench custom

v09:  ## Multimodal fusion, symplectic RL and wavefield swarm tracks
	@$(RCWE) bench v09

predict:  ## Legacy one-shot predictor on the sample grid log
	@$(PYTHON) -m rcwe.predict_cli --input examples/data/sample_grid_telemetry.csv

benchmarks:  ## Run every suite and rebuild docs/BENCHMARKS.md from the results
	@$(PYTHON) scripts/build_benchmarks_doc.py

clean:  ## Remove generated files and caches
	@rm -rf rcwe_output rcwe_quickstart rcwe_demo benchmarks_data build dist .pytest_cache
	@find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +
	@echo "Cleaned generated files."
