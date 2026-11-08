VENV_PYTHON := $(HOME)/Downloads/.venv/bin/python
PYTEST := $(HOME)/Downloads/.venv/bin/pytest
BENCH := $(HOME)/Downloads/.venv/bin/rcwe-bench

.PHONY: fast test bench clean-temp

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
