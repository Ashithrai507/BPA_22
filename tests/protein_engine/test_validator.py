from __future__ import annotations

from protein_engine.sequence.validator import dedupe_by_accession, split_valid, validate


def test_validate_ok() -> None:
    result = validate({"accession": "P1", "sequence": "M" * 30})
    assert result["ok"] is True
    assert result["issues"] == []


def test_validate_too_short() -> None:
    result = validate({"accession": "P1", "sequence": "M" * 10})
    assert result["ok"] is False
    assert "too_short" in result["issues"]


def test_validate_non_standard_residues() -> None:
    result = validate({"accession": "P1", "sequence": "M" * 30 + "XZ"})
    assert result["ok"] is False
    assert any(issue.startswith("non_standard:") for issue in result["issues"])


def test_dedupe_by_accession_keeps_first() -> None:
    proteins = [{"accession": "P1"}, {"accession": "P2"}, {"accession": "P1"}]
    assert [p["accession"] for p in dedupe_by_accession(proteins)] == ["P1", "P2"]


def test_split_valid_partitions() -> None:
    good = {"accession": "P1", "sequence": "M" * 30}
    bad = {"accession": "P2", "sequence": "Z" * 30}
    valid, excluded = split_valid([good, bad, good])
    assert [p["accession"] for p in valid] == ["P1"]
    assert [p["accession"] for p in excluded] == ["P2"]
    assert "excluded_reasons" in excluded[0]
