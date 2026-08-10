from __future__ import annotations

from conftest import FakeResponse, FakeTransport, json_response

from protein_engine.retrieval.api_cache import ApiCache
from protein_engine.retrieval.ncbi_client import fetch_proteins


def test_ncbi_fetch_parses_fasta(tmp_path) -> None:
    fasta = ">sp|P37476|SP0A_BACSU Stage 0 sporulation protein A\nMNNKILV\n>gi|123|hypothetical\nACDEFG\n"
    transport = FakeTransport([json_response({"esearchresult": {"idlist": ["1", "2"]}}), FakeResponse(fasta)])
    cache = ApiCache(tmp_path / "http.db", transport=transport, rate_limit=0.0, backoff=[])
    proteins = fetch_proteins(224308, cache)
    assert len(proteins) == 2
    assert proteins[0]["accession"] == "P37476"
    assert proteins[0]["sequence"] == "MNNKILV"
    assert proteins[0]["reviewed"] is False
    assert proteins[0]["evidence"] == "unreviewed"
    assert "txid224308[Organism]" in transport.calls[0]["params"]["term"]


def test_ncbi_fetch_empty_when_no_ids(tmp_path) -> None:
    transport = FakeTransport([json_response({"esearchresult": {"idlist": []}})])
    cache = ApiCache(tmp_path / "http.db", transport=transport, rate_limit=0.0, backoff=[])
    assert fetch_proteins(224308, cache) == []
