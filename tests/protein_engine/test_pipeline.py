from __future__ import annotations

import json

from conftest import FakeTransport, json_response

from protein_engine.pipeline import load_curated, run


def _uniprot_result(accession: str, gene: str, length: int = 100) -> dict:
    return {
        "primaryAccession": accession,
        "proteinDescription": {"recommendedName": {"fullName": {"value": f"{gene} protein"}}},
        "genes": [{"geneName": {"value": gene}}],
        "sequence": {"value": "M" * length, "length": length},
        "comments": [],
        "subcellularLocations": [],
        "uniProtKBCrossReferences": [],
        "keywords": [],
    }


def test_load_curated_picks_existing_files(tmp_path) -> None:
    (tmp_path / "essential_genes.csv").write_text("gene\nspoA\n")
    tables = load_curated(tmp_path)
    assert "essential" in tables
    assert "virulence" not in tables
    assert tables["essential"].matches({"gene": "spoA", "accession": "x"}) is True


def test_run_end_to_end(tmp_path) -> None:
    reference_dir = tmp_path / "ref"
    reference_dir.mkdir()
    (reference_dir / "essential_genes.csv").write_text("gene\nspoA\n")
    results = [_uniprot_result("P37476", "spoA", 405)]
    results += [_uniprot_result(f"P0A0{str(i).zfill(2)}", f"gene{i}", 300) for i in range(11)]
    pdb_responses = [json_response([{"struct_id": "1FSE"}])] + [json_response([]) for _ in range(11)]
    transport = FakeTransport(
        [
            json_response({"esearchresult": {"idlist": ["224308"]}}),
            json_response({"result": {"224308": {"scientificname": "Bacillus subtilis"}}}),
            json_response({"results": results}),
            *pdb_responses,
        ]
    )
    payload = run(
        "Bacillus subtilis",
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        reference_dir=str(reference_dir),
        transport=transport,
        rate_limit=0.0,
    )
    assert payload["species"] == "Bacillus subtilis"
    assert payload["taxonomy_id"] == 224308
    assert payload["source"] == "uniprot"
    assert payload["selected_proteins"][0]["accession"] == "P37476"
    assert payload["selected_proteins"][0]["score"] == 0.55
    assert payload["output_dir"]
    out_dir = tmp_path / "out" / "bacillus_subtilis"
    assert (out_dir / "proteins.json").exists()
    assert (out_dir / "sequences.fasta").exists()
    stored = json.loads((out_dir / "proteins.json").read_text())
    assert stored["species"] == "Bacillus subtilis"
    assert stored["selected_proteins"][0]["sequence"].startswith("M")
