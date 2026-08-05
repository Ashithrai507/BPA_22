PYTHON ?= python
VENV_DIR ?= .venv
VENV_PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_PYTHON) -m pip
PYTEST = $(VENV_PYTHON) -m pytest
RUFF = $(VENV_PYTHON) -m ruff

.PHONY: help setup install train test lint format run-ui ci-upload-artifact clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Create virtualenv and install dependencies
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

install: ## Install dependencies into the active environment
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

train: ## Train the model bundle (requires data/dataset present)
	$(VENV_PYTHON) scripts/train_model.py

test: ## Run the test suite
	$(PYTEST) -q

lint: ## Lint and check formatting
	$(RUFF) check .
	$(RUFF) format --check .

format: ## Auto-format code with ruff
	$(RUFF) format .

run-ui: ## Launch the desktop UI
	$(VENV_PYTHON) scripts/bacteria_ui.py

ci-upload-artifact: ## Upload the trained artifact so CI can run full tests (needs dataset + gh auth)
	gh release view model-artifact >/dev/null 2>&1 || gh release create model-artifact --title "Trained model artifact" --notes "Seeded for CI. Regenerate with 'make train'."
	gh release upload model-artifact artifacts/bacteria_models.joblib --clobber

clean: ## Remove build and cache artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
