from __future__ import annotations

from server.models import PredictResponse, ProteinRequest, SpeciesInfo


def test_predict_response_optional_fields() -> None:
    resp = PredictResponse(id=1, predicted_species="E. coli")
    assert resp.id == 1
    assert resp.colonies is None
    assert resp.morphology is None


def test_protein_request_defaults() -> None:
    req = ProteinRequest(species="Bacillus subtilis")
    assert req.top_n == 10


def test_species_info_fields() -> None:
    sp = SpeciesInfo(
        name="E. coli",
        organism_type="bacteria",
        gram="gram_negative",
        shape="bacilli",
        group="gram_negative_bacilli",
    )
    assert sp.name == "E. coli"
