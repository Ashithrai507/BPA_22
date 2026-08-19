from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from server.main import app

client = TestClient(app)


def test_list_species() -> None:
    resp = client.get("/api/species")
    assert resp.status_code == 200
    species = resp.json()
    assert len(species) == 10
    names = {s["name"] for s in species}
    assert "Bacillus subtilis" in names
    assert "Escherichia coli" in names


FAKE_PROTEIN_RESULT = {
    "species": "Bacillus subtilis",
    "taxonomy_id": 224308,
    "resolved_name": "Bacillus subtilis",
    "source": "uniprot",
    "selected_proteins": [
        {
            "accession": "P37476",
            "protein_name": "spoA protein",
            "gene": "spoA",
            "length": 405,
            "reviewed": True,
            "score": 0.55,
            "rank": 1,
            "has_structure": True,
        }
    ],
    "excluded": [],
}


@patch("protein_engine.pipeline.run", return_value=FAKE_PROTEIN_RESULT)
def test_rank_proteins(mock_run) -> None:
    resp = client.post(
        "/api/proteins", json={"species": "Bacillus subtilis", "top_n": 10}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["species"] == "Bacillus subtilis"
    assert len(body["selected_proteins"]) == 1
    assert body["selected_proteins"][0]["accession"] == "P37476"


@patch("protein_engine.pipeline.run", return_value=FAKE_PROTEIN_RESULT)
def test_batch_proteins(mock_run) -> None:
    resp = client.post(
        "/api/proteins/batch",
        json=["Bacillus subtilis", "Escherichia coli"],
        params={"top_n": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[0]["species"] == "Bacillus subtilis"
    assert body[1]["species"] == "Bacillus subtilis"


@patch("protein_engine.pipeline.run", side_effect=Exception("boom"))
def test_batch_proteins_handles_failure(mock_run) -> None:
    resp = client.post(
        "/api/proteins/batch",
        json=["Bacillus subtilis"],
        params={"top_n": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["source"] == "error"
    assert body[0]["selected_proteins"] == []
