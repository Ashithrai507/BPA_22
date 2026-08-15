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
