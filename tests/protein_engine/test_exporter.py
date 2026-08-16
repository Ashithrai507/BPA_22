from __future__ import annotations

import json

from protein_engine.sequence.exporter import write


def _protein() -> dict:
    return {
        "rank": 1,
        "accession": "P37476",
        "gene": "spo0A",
        "protein_name": "Spo0A",
        "length": 60,
        "sequence": "M" * 60,
        "score": 0.55,
        "score_breakdown": {"essential": 1.0, "virulence": 0.0, "resistance": 0.0, "evidence": 1.0, "structure": 0.0},
        "evidence": "reviewed",
        "has_structure": False,
        "subcellular_location": "Cytoplasm",
        "function": "response regulator",
    }


def test_write_produces_contract(tmp_path) -> None:
    proteins = [_protein()]
    out = write(
        "Bacillus subtilis",
        "Bacillus subtilis subsp. subtilis",
        224308,
        proteins,
        [],
        tmp_path,
        source="uniprot",
    )
    assert out == tmp_path / "bacillus_subtilis_subsp_subtilis"
    payload = json.loads((out / "proteins.json").read_text())
    assert payload["species"] == "Bacillus subtilis"
    assert payload["taxonomy_id"] == 224308
    assert payload["resolved_name"] == "Bacillus subtilis subsp. subtilis"
    assert payload["source"] == "uniprot"
    assert payload["query_timestamp"]
    assert payload["excluded"] == []
    assert payload["selected_proteins"][0]["accession"] == "P37476"
    assert "sequence" in payload["selected_proteins"][0]
    fasta = (out / "sequences.fasta").read_text()
    assert fasta.startswith(">P37476|spo0A|Spo0A|Bacillus subtilis subsp. subtilis\n")
