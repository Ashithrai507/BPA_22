from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import APIRouter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.config import ORGANISM_METADATA  # noqa: E402

from ..dependencies import get_db_path  # noqa: E402
from ..models import ProteinRequest, ProteinResponse, SpeciesInfo  # noqa: E402

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["proteins"])


@router.get("/species", response_model=list[SpeciesInfo])
def list_species() -> list[SpeciesInfo]:
    return [
        SpeciesInfo(
            name=name,
            organism_type=m["organism_type"],
            gram=m["gram_label"],
            shape=m["shape_label"],
            group=m["taxonomy_group"],
        )
        for name, m in ORGANISM_METADATA.items()
    ]


@router.post("/proteins", response_model=ProteinResponse)
def rank_proteins(request: ProteinRequest) -> ProteinResponse:
    from protein_engine.pipeline import run

    result = run(request.species, top_n=request.top_n)
    return ProteinResponse(
        species=result["species"],
        taxonomy_id=result["taxonomy_id"],
        resolved_name=result["resolved_name"],
        source=result["source"],
        selected_proteins=result["selected_proteins"],
        excluded=result.get("excluded", []),
    )


@router.post("/proteins/batch", response_model=list[ProteinResponse])
async def rank_proteins_batch(species_list: list[str], top_n: int = 10) -> list[ProteinResponse]:
    from protein_engine.pipeline import run

    results = []
    for species in species_list:
        try:
            result = run(species, top_n=top_n)
            results.append(
                ProteinResponse(
                    species=result["species"],
                    taxonomy_id=result["taxonomy_id"],
                    resolved_name=result["resolved_name"],
                    source=result["source"],
                    selected_proteins=result["selected_proteins"],
                    excluded=result.get("excluded", []),
                )
            )
        except Exception:
            logger.warning("Batch analysis failed for species=%s", species, exc_info=True)
            results.append(
                ProteinResponse(
                    species=species,
                    taxonomy_id=0,
                    resolved_name=species,
                    source="error",
                    selected_proteins=[],
                    excluded=[],
                )
            )
    return results


@router.get("/proteins/history")
def protein_history(limit: int = 50, offset: int = 0) -> dict:
    from ..db import list_protein_analyses

    db_path = get_db_path()
    return list_protein_analyses(db_path, limit=limit, offset=offset)
