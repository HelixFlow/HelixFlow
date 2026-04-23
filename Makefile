# HelixFlow Makefile — thin task runner for common dev / CI workflows.

PYTHON        ?= python3
PIP           ?= pip
RUFF          ?= ruff
PYTEST        ?= pytest
COMPOSE       ?= docker compose

.PHONY: help install install-dev lint fmt test test-unit test-integration test-concurrent \
        web-audit clean

help:
	@echo "Common targets:"
	@echo "  install           — install runtime deps"
	@echo "  install-dev       — install runtime + test deps"
	@echo "  lint              — ruff check ."
	@echo "  fmt               — ruff format ."
	@echo "  test              — pytest (unit + api + smoke + concurrent)"
	@echo "  test-unit         — pytest tests/unit"
	@echo "  test-integration  — pytest tests/integration (requires docker)"
	@echo "  test-concurrent   — pytest tests/concurrent"
	@echo "  web-audit         — run scripts/audit-web-submodule.sh"
	@echo "  clean             — remove build / coverage artifacts"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

lint:
	$(RUFF) check .

fmt:
	$(RUFF) format .

test:
	$(PYTEST) tests/unit tests/api tests/smoke tests/concurrent tests/persistence -m "not integration" -v

test-unit:
	$(PYTEST) tests/unit -v

test-integration:
	$(PYTEST) tests/integration -m integration -v

test-concurrent:
	$(PYTEST) tests/concurrent -v

web-audit:
	@bash scripts/audit-web-submodule.sh

clean:
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov __pycache__
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
