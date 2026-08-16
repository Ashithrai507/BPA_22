from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "run_protein_pipeline.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )


def test_cli_help_exits_zero() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "--species" in result.stdout


def test_cli_missing_species_and_image_exits_two() -> None:
    result = _run()
    assert result.returncode == 2
    assert "either --species or --image is required" in result.stderr


def test_cli_top_n_zero_rejected() -> None:
    result = _run("--species", "Bacillus subtilis", "--top-n", "0")
    assert result.returncode == 2
