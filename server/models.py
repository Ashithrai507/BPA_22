from __future__ import annotations

from pydantic import BaseModel


class PredictResponse(BaseModel):
    id: int
    created_at: str | None = None
    organism_type: str | None = None
    predicted_species: str | None = None
    gram: str | None = None
    total_colonies: int | None = None
    dominant_shape: str | None = None
    confidence: float | None = None
    colonies: list[dict] | None = None
    morphology: dict | None = None


class HistoryItem(BaseModel):
    id: int
    created_at: str | None = None
    filename: str
    predicted_species: str | None = None
    confidence: float | None = None
    protein_status: str | None = None


class HistoryResponse(BaseModel):
    total: int
    items: list[HistoryItem]


class ProteinRequest(BaseModel):
    species: str
    top_n: int = 10


class ProteinResponse(BaseModel):
    species: str
    taxonomy_id: int
    resolved_name: str
    source: str
    selected_proteins: list[dict]
    excluded: list[dict]


class SpeciesInfo(BaseModel):
    name: str
    organism_type: str
    gram: str
    shape: str
    group: str
