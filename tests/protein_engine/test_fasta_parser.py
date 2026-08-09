from __future__ import annotations

from protein_engine.sequence.fasta_parser import parse, to_fasta


def test_parse_multi_record() -> None:
    text = ">P37476|spo0A|Stage 0 sporulation protein A\nMNNKILV\n>P0A|gene2\nACDEFG\n"
    records = parse(text)
    assert len(records) == 2
    assert records[0]["accession"] == "P37476"
    assert records[0]["header"] == "P37476|spo0A|Stage 0 sporulation protein A"
    assert records[0]["sequence"] == "MNNKILV"
    assert records[1]["accession"] == "P0A"
    assert records[1]["sequence"] == "ACDEFG"


def test_parse_ncbi_style_accession() -> None:
    records = parse(">sp|P37476|SP0A_BACSU Stage 0 sporulation protein A\nMNNK\n")
    assert records[0]["accession"] == "P37476"


def test_to_fasta_wraps_and_builds_header() -> None:
    proteins = [{"accession": "P37476", "gene": "spo0A", "protein_name": "Spo0A", "sequence": "M" * 130}]
    text = to_fasta(proteins, "Bacillus subtilis")
    assert text.startswith(">P37476|spo0A|Spo0A|Bacillus subtilis\n")
    lines = text.splitlines()
    assert len(lines) == 4  # header + 60 + 60 + 10
    assert len(lines[1]) == 60
