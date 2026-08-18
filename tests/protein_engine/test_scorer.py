from __future__ import annotations

import pytest

from protein_engine.ranking._common import CuratedTable
from protein_engine.ranking.scorer import compute_score, preselect, rank, score_terms


def _tables() -> dict:
    return {
        "essential": CuratedTable("essential", frozenset({"spoa"}), frozenset()),
        "virulence": CuratedTable("virulence", frozenset(), frozenset()),
        "resistance": CuratedTable("resistance", frozenset(), frozenset()),
    }


def _protein(accession: str, gene: str, reviewed: bool) -> dict:
    return {"accession": accession, "gene": gene, "reviewed": reviewed, "sequence": "M" * 100}


def test_score_terms_essential_evidence_structure() -> None:
    terms = score_terms(_protein("P37476", "spoA", True), curated=_tables(), has_structure=True)
    assert terms == {"essential": 1.0, "virulence": 0.0, "resistance": 0.0, "evidence": 1.0, "structure": 1.0}


def test_score_terms_unreviewed_evidence_is_half() -> None:
    terms = score_terms(_protein("P0A", "x", False), curated=_tables())
    assert terms["evidence"] == 0.5


def test_compute_score_uses_weights() -> None:
    terms = {"essential": 1.0, "virulence": 0.0, "resistance": 0.0, "evidence": 1.0, "structure": 1.0}
    assert compute_score(terms) == pytest.approx(0.55)


def test_rank_orders_and_injects_breakdown() -> None:
    proteins = [_protein("P37476", "spoA", True), _protein("P0A", "x", False)]
    ranked = rank(proteins, structure_flags={"P37476": True}, curated=_tables(), top_n=2)
    assert ranked[0]["accession"] == "P37476"
    assert ranked[0]["score"] == 0.55
    assert ranked[0]["score_breakdown"]["essential"] == 1.0
    assert ranked[1]["score"] == round(0.075, 3)


def test_rank_honors_top_n() -> None:
    proteins = [_protein("P1", "a", True), _protein("P2", "b", True)]
    ranked = rank(proteins, top_n=1)
    assert len(ranked) == 1


def test_preselect_excludes_structure_term() -> None:
    proteins = [_protein("P37476", "spoA", True)]
    selected = preselect(proteins, curated=_tables(), limit=10)
    assert selected[0]["score"] == 0.45  # 0.30 essential + 0.15 evidence, no structure


def test_rank_tie_breaks_by_accession() -> None:
    p1 = _protein("P_B", "x", True)
    p2 = _protein("P_A", "y", True)
    ranked = rank([p1, p2], curated=_tables(), top_n=2)
    assert ranked[0]["accession"] == "P_A"
    assert ranked[1]["accession"] == "P_B"


def test_preselect_returns_descending_order() -> None:
    low = _protein("P_LOW", "low", False)
    high = _protein("P_HIGH", "spoa", True)
    selected = preselect([low, high], curated=_tables(), limit=10)
    assert selected[0]["accession"] == "P_HIGH"
    assert selected[1]["accession"] == "P_LOW"
    assert selected[0]["score"] >= selected[1]["score"]


def test_score_terms_virulence_and_resistance_combined() -> None:
    tables = {
        "essential": CuratedTable("essential", frozenset(), frozenset()),
        "virulence": CuratedTable("virulence", frozenset({"toxa"}), frozenset()),
        "resistance": CuratedTable("resistance", frozenset({"teta"}), frozenset()),
    }
    terms = score_terms(_protein("P1", "toxA", True), curated=tables)
    assert terms["virulence"] == 1.0
    assert terms["resistance"] == 0.0
    terms2 = score_terms(_protein("P2", "tetA", True), curated=tables)
    assert terms2["virulence"] == 0.0
    assert terms2["resistance"] == 1.0
    both = _protein("P3", "toxA", True)
    both["gene"] = "toxA"
    terms3 = score_terms(both, curated=tables)
    assert terms3["virulence"] == 1.0
    assert terms3["resistance"] == 0.0
