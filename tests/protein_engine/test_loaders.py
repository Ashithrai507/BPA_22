from __future__ import annotations

from protein_engine.ranking.essential import load_essential_genes
from protein_engine.ranking.resistance import load_resistance_genes
from protein_engine.ranking.virulence import load_virulence_genes


def _write(path, text: str) -> None:
    path.write_text(text)


def test_essential_matches_gene_or_accession(tmp_path) -> None:
    p = tmp_path / "essential.csv"
    _write(p, "gene,accession\nspoA,\n,P37476\n")
    table = load_essential_genes(p)
    assert table.name == "essential"
    assert table.matches({"gene": "spoA", "accession": "X"}) is True
    assert table.matches({"gene": "zap", "accession": "p37476"}) is True
    assert table.matches({"gene": "zap", "accession": "X"}) is False


def test_virulence_and_resistance_share_columns(tmp_path) -> None:
    vp = tmp_path / "vf.csv"
    rp = tmp_path / "card.csv"
    _write(vp, "gene_symbol\nhlyA\n")
    _write(rp, "gene\ntetA\n")
    assert load_virulence_genes(vp).matches({"gene": "hlya", "accession": "X"}) is True
    assert load_resistance_genes(rp).matches({"gene": "teta", "accession": "X"}) is True


def test_missing_gene_columns_produce_empty_table(tmp_path) -> None:
    p = tmp_path / "empty.csv"
    _write(p, "gene\n")
    table = load_essential_genes(p)
    assert table.matches({"gene": "x", "accession": "y"}) is False
