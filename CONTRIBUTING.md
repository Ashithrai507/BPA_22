# Contributing to BPA-22

Thanks for contributing! This guide covers how to set up a working dev environment,
the workflow we use, and how to keep CI green.

## 1. Prerequisites

- Python 3.11+
- `git`
- (Recommended) `make`

## 2. Environment setup

```bash
git clone https://github.com/Ashithrai507/BPA_22.git
cd BPA_22
make setup        # creates .venv, installs runtime + dev deps
```

`make setup` is equivalent to:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e ".[dev]"
```

### Optional: Docker dev container

A `.devcontainer/` is provided for environment parity on Linux/macOS hosts
(CLI, training, and testing only — the PyQt5 GUI needs a real display).
It mounts your local `Bacteria dataset/` and `artifacts/` as volumes.

## 3. Getting the dataset and model artifact

The **dataset** (`Bacteria dataset/`, ~302 MB) and the **trained model**
(`artifacts/bacteria_models.joblib`, ~662 MB) are **not stored in git**.
They are shared manually (shared drive / direct copy).

1. Copy the `Bacteria dataset/` folder into the repo root.
2. Run `make train` to build the model bundle (or copy an existing
   `artifacts/bacteria_models.joblib`).

### Overriding paths per developer

Each developer may keep the dataset/model somewhere else. Copy `.env.example` to
`.env` and set `BACTERIA_DATASET_ROOT`, `BACTERIA_ARTIFACT_DIR`, or
`BACTERIA_MODEL_PATH`. Example:

```bash
BACTERIA_DATASET_ROOT=/Volumes/shared/Bacteria_dataset
```

## 4. Common commands

| Task            | Command            |
| --------------- | ------------------ |
| Train model     | `make train`       |
| Run tests       | `make test`        |
| Lint + format check | `make lint`    |
| Auto-format     | `make format`      |
| Launch UI       | `make run-ui`      |

## 5. Code quality

- **Formatting & linting**: `ruff` (config in `pyproject.toml`). Run
  `make lint` before pushing; `make format` auto-fixes.
- **Pre-commit** (optional): install once with
  `pre-commit install` to run ruff + sanity checks on every commit.

## 6. Workflow

1. Create a feature branch off `main`:
   `git checkout -b feature/your-change`.
2. Make your changes, run `make lint` and `make test`.
3. Push and open a pull request using the template.
4. CI runs lint + tests on your PR. `main` is protected: at least one approving
   review and green CI are required.

## 7. How CI gets the trained model

CI cannot train (it has no dataset), so it needs a **seeded artifact**:

- CI first restores the artifact from the Actions **cache**.
- On a cache miss it downloads it from the GitHub release named
  `model-artifact`.

If you regenerate the model (new feature/retrain), reseed it from a machine that
has the dataset:

```bash
make ci-upload-artifact   # requires gh auth + dataset present
```

If the artifact is missing entirely, CI fails with a clear message (it never
silently skips tests).

## 8. Tests

`tests/test_output_contract.py` validates the basic/advanced JSON output contracts.
Tests `skip` with a visible reason when the dataset or model artifact is absent
locally — they never pass vacuously.
