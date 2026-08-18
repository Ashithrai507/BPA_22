from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.config import MODEL_PATH  # noqa: E402
from bacteria_assistant.inference import load_models  # noqa: E402

_models: dict | None = None


def get_models() -> dict:
    global _models
    if _models is None:
        model_path = PROJECT_ROOT / MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        _models = load_models(model_path)
    return _models


def get_db_path() -> Path:
    return PROJECT_ROOT / "data" / "bpa.db"
