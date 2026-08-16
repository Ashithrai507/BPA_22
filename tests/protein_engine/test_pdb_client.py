from __future__ import annotations

import pytest
from conftest import FakeResponse, FakeTransport, json_response

from protein_engine.config import PDB_UNIPROT
from protein_engine.retrieval.api_cache import ApiCache, ApiError
from protein_engine.retrieval.pdb_client import has_structure


def test_has_structure_true(tmp_path) -> None:
    transport = FakeTransport([json_response([{"struct_id": "1FSE"}])])
    cache = ApiCache(tmp_path / "http.db", transport=transport, rate_limit=0.0, backoff=[])
    assert has_structure("P37476", cache) is True


def test_has_structure_false_on_404(tmp_path) -> None:
    transport = FakeTransport([FakeResponse("", status_code=404)])
    cache = ApiCache(tmp_path / "http.db", transport=transport, rate_limit=0.0, backoff=[])
    assert has_structure("P0A1B2", cache) is False


def test_has_structure_404_caches_negative_marker(tmp_path) -> None:
    transport = FakeTransport([FakeResponse("", status_code=404)])
    cache = ApiCache(tmp_path / "http.db", transport=transport, rate_limit=0.0, backoff=[])
    assert has_structure("P0A1B2", cache) is False
    assert cache.get(f"{PDB_UNIPROT}/P0A1B2") == "null"
    assert len(transport.calls) == 1
    assert has_structure("P0A1B2", cache) is False


def test_has_structure_propagates_other_errors(tmp_path) -> None:
    transport = FakeTransport([FakeResponse("", status_code=500), FakeResponse("", status_code=500)])
    cache = ApiCache(tmp_path / "http.db", transport=transport, rate_limit=0.0, backoff=[0.0])
    with pytest.raises(ApiError) as excinfo:
        has_structure("P0A1B2", cache)
    assert excinfo.value.status_code == 500
