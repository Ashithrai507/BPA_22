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


def test_validate_non_standard_residues_exact_chars() -> None:
    result = validate({"accession": "P1", "sequence": "M" * 30 + "XZ"})
    assert result["ok"] is False
    issues = [i for i in result["issues"] if i.startswith("non_standard:")]
    assert len(issues) == 1
    chars = issues[0].split(":")[1]
    assert sorted(chars) == ["X", "Z"]


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


def test_dedupe_none_accessions_treated_as_single_group() -> None:
    p1 = {"accession": None, "sequence": "M" * 30}
    p2 = {"accession": None, "sequence": "M" * 30}
    result = dedupe_by_accession([p1, p2])
    assert len(result) == 1
