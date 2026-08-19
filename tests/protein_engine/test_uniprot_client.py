from __future__ import annotations

from conftest import FakeTransport, json_response

from protein_engine.retrieval.api_cache import ApiCache
from protein_engine.retrieval.uniprot_client import fetch_proteins, normalize


def _cache(path, transport):
    return ApiCache(path, transport=transport, rate_limit=0.0, backoff=[])


def _result(accession: str, gene: str) -> dict:
    return {
        "primaryAccession": accession,
        "proteinDescription": {"recommendedName": {"fullName": {"value": f"{gene} protein"}}},
        "genes": [{"geneName": {"value": gene}}],
        "sequence": {"value": "M" * 100, "length": 100},
        "comments": [
            {"commentType": "FUNCTION", "texts": [{"value": "does things"}]},
            {"commentType": "SUBCELLULAR LOCATION", "subcellularLocations": [{"location": {"value": "Cytoplasm"}}]},
        ],
        "uniProtKBCrossReferences": [{"database": "PDB", "id": "1FSE"}],
        "keywords": [{"name": "Sporulation"}],
    }


def test_normalize_maps_fields() -> None:
    rec = normalize(_result("P37476", "spo0A"), reviewed=True)
    assert rec["accession"] == "P37476"
    assert rec["gene"] == "spo0A"
    assert rec["protein_name"] == "spo0A protein"
    assert rec["sequence"] == "M" * 100
    assert rec["length"] == 100
    assert rec["reviewed"] is True
    assert rec["evidence"] == "reviewed"
    assert rec["subcellular_location"] == "Cytoplasm"
    assert rec["function"] == "does things"
    assert rec["pdb_structures"] == ["1FSE"]
    assert rec["keywords"] == ["Sporulation"]


def test_fetch_proteins_reviewed_first(tmp_path) -> None:
    transport = FakeTransport([json_response({"results": [_result("P37476", "spo0A")]})])
    cache = _cache(tmp_path / "http.db", transport)
    proteins = fetch_proteins(224308, cache, reviewed_min_count=1)
    assert len(proteins) == 1
    assert proteins[0]["accession"] == "P37476"
    assert "reviewed:true" in transport.calls[0]["params"]["query"]


def test_fetch_proteins_falls_back_to_unreviewed_when_too_few(tmp_path) -> None:
    transport = FakeTransport(
        [json_response({"results": [_result("P0A", "x")]}), json_response({"results": [_result("P0A", "x")]})]
    )
    cache = _cache(tmp_path / "http.db", transport)
    proteins = fetch_proteins(224308, cache, reviewed_min_count=10)
    assert proteins[0]["evidence"] == "unreviewed"
    assert "reviewed:false" in transport.calls[1]["params"]["query"]


def test_fetch_proteins_walks_pagination(tmp_path) -> None:
    first = {"results": [_result("P1", "a")], "nextLink": "https://rest.uniprot.org/uniprotkb/search?cursor=A"}
    second = {"results": [_result("P2", "b")]}
    transport = FakeTransport([json_response(first), json_response(second)])
    cache = _cache(tmp_path / "http.db", transport)
    proteins = fetch_proteins(224308, cache, reviewed_min_count=2)
    assert [p["accession"] for p in proteins] == ["P1", "P2"]
    assert transport.calls[1]["params"] is None
