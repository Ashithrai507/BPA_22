from __future__ import annotations

from protein_engine.config import DEFAULT_TOP_N, PRESELECT_LIMIT, RANKING_WEIGHTS
from protein_engine.ranking._common import CuratedTable


def score_terms(
    protein: dict,
    curated: dict | None = None,
    has_structure: bool = False,
    weights: dict | None = None,
) -> dict:
    weights = weights or RANKING_WEIGHTS
    curated = curated or {}
    essential_table: CuratedTable | None = curated.get("essential")
    virulence_table: CuratedTable | None = curated.get("virulence")
    resistance_table: CuratedTable | None = curated.get("resistance")
    return {
        "essential": 1.0 if essential_table and essential_table.matches(protein) else 0.0,
        "virulence": 1.0 if virulence_table and virulence_table.matches(protein) else 0.0,
        "resistance": 1.0 if resistance_table and resistance_table.matches(protein) else 0.0,
        "evidence": 1.0 if protein.get("reviewed") else 0.5,
        "structure": 1.0 if has_structure else 0.0,
    }


def compute_score(terms: dict, weights: dict | None = None) -> float:
    weights = weights or RANKING_WEIGHTS
    return sum(weights[key] * terms[key] for key in weights)


def _scored(
    proteins: list[dict],
    curated: dict | None,
    structure_flags: dict,
    weights: dict | None = None,
) -> list[dict]:
    flags = structure_flags or {}
    scored = []
    for protein in proteins:
        terms = score_terms(
            protein,
            curated=curated,
            has_structure=flags.get(protein.get("accession"), False),
            weights=weights,
        )
        scored.append({**protein, "score": round(compute_score(terms, weights), 3), "score_breakdown": terms})
    scored.sort(key=lambda item: (-item["score"], item["accession"]))
    return scored


def preselect(proteins: list[dict], curated: dict | None = None, limit: int = PRESELECT_LIMIT) -> list[dict]:
    return _scored(proteins, curated, {})[:limit]


def rank(
    proteins: list[dict],
    structure_flags: dict | None = None,
    curated: dict | None = None,
    weights: dict | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict]:
    return _scored(proteins, curated, structure_flags, weights)[:top_n]
