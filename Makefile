VENV_PYTHON := $(HOME)/Downloads/.venv/bin/python
PYTEST := $(HOME)/Downloads/.venv/bin/pytest
BENCH := $(HOME)/Downloads/.venv/bin/rcwe-bench

.PHONY: fast test bench dashboard extreme clean-temp

fast:
	@$(VENV_PYTHON) -m compileall -q rcwe tests
	@$(PYTEST) -q tests/test_invariance.py tests/test_plasticity_and_mdo.py

test:
	@$(VENV_PYTHON) -m compileall -q rcwe tests
	@$(PYTEST) -v

bench:
	@$(BENCH)

clean-temp:
	@rm -rf $(HOME)/Downloads/rcwe-clean-* $(HOME)/Downloads/rcwe-v02* $(HOME)/Downloads/rcwe-v03-*
	@echo Cleaned old temporary directories.

extreme:
	@$(VENV_PYTHON) -m rcwe.diagnostics.extreme_bench

dashboard:
	@$(VENV_PYTHON) -m rcwe.diagnostics.dashboard_v05

realworld:
	@$(VENV_PYTHON) -m rcwe.benchmarks.real_world_suite

predict:
	@$(VENV_PYTHON) -m rcwe.predict_cli --input sample_grid_telemetry.csv

serve:
	@$(VENV_PYTHON) -m rcwe.server.live_server

phase2:
	@$(VENV_PYTHON) -m rcwe.diagnostics.phase2_v07

train-corpus:
	@$(VENV_PYTHON) -m rcwe.lm.corpus_trainer

custom-data:
	@$(VENV_PYTHON) -m rcwe.ingestion.custom_connector

v09:
	@$(VENV_PYTHON) -m rcwe.v09_runner

ui:
	@$(VENV_PYTHON) -m rcwe.ui.mission_control
