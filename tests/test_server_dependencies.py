from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from server.dependencies import get_db_path


def test_get_db_path() -> None:
    path = get_db_path()
    assert path.name == "bpa.db"
    assert path.parent == PROJECT_ROOT / "data"
