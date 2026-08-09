from __future__ import annotations

import pytest
from conftest import FakeTransport, json_response

from protein_engine.retrieval.api_cache import ApiCache
from protein_engine.taxonomy.cache import TaxonomyCache
from protein_engine.taxonomy.resolver import ResolutionError, normalize_name, resolve


def _cache(path, transport, **kwargs):
    return ApiCache(path, transport=transport, rate_limit=0.0, backoff=[], **kwargs)


def test_normalize_name() -> None:
    assert normalize_name("  Bacillus   subtilis ") == "bacillus subtilis"


def test_resolve_via_ncbi_and_cache(tmp_path) -> None:
    transport = FakeTransport(
        [
            json_response({"esearchresult": {"idlist": ["224308"]}}),
            json_response({"result": {"224308": {"scientificname": "Bacillus subtilis subsp. subtilis"}}}),
        ]
    )
    cache = _cache(tmp_path / "http.db", transport)
    tcache = TaxonomyCache(tmp_path / "taxonomy.db")
    assert resolve("Bacillus subtilis", cache, tcache) == {
        "taxonomy_id": 224308,
        "resolved_name": "Bacillus subtilis subsp. subtilis",
    }
    assert tcache.get("bacillus subtilis") == {
        "taxonomy_id": 224308,
        "resolved_name": "Bacillus subtilis subsp. subtilis",
    }


def test_resolve_second_call_hits_taxonomy_cache(tmp_path) -> None:
    tcache = TaxonomyCache(tmp_path / "taxonomy.db")
    tcache.set("bacillus subtilis", {"taxonomy_id": 224308, "resolved_name": "Bacillus subtilis"})
    cache = _cache(tmp_path / "http.db", FakeTransport([]))
    assert resolve("  Bacillus subtilis  ", cache, tcache) == {
        "taxonomy_id": 224308,
        "resolved_name": "Bacillus subtilis",
    }


def test_resolve_raises_when_unknown(tmp_path) -> None:
    transport = FakeTransport([json_response({"esearchresult": {"idlist": []}})])
    cache = _cache(tmp_path / "http.db", transport)
    with pytest.raises(ResolutionError):
        resolve("Unknownus mysteryii", cache)


def test_resolve_raises_on_empty(tmp_path) -> None:
    cache = _cache(tmp_path / "http.db", FakeTransport([]))
    with pytest.raises(ResolutionError):
        resolve("   ", cache)
