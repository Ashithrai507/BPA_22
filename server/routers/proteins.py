from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.config import ORGANISM_METADATA

from ..models import ProteinRequest, ProteinResponse, SpeciesInfo

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
