from __future__ import annotations

import json

from protein_engine.config import NCBI_ESEARCH, NCBI_ESUMMARY
from protein_engine.retrieval.api_cache import ApiCache
from protein_engine.taxonomy.cache import TaxonomyCache


class ResolutionError(ValueError):
    pass


def normalize_name(species: str) -> str:
    return " ".join(str(species).strip().lower().split())


def resolve(species: str, cache: ApiCache, taxonomy_cache: TaxonomyCache | None = None) -> dict:
    name = normalize_name(species)
    if not name:
        raise ResolutionError("empty species name")
    if taxonomy_cache is not None:
        hit = taxonomy_cache.get(name)
        if hit is not None:
            return hit
    body = cache.fetch(NCBI_ESEARCH, params={"db": "taxonomy", "term": f'"{name}"[All Names]', "retmode": "json"})
    id_list = ((json.loads(body).get("esearchresult") or {}).get("idlist")) or []
    if not id_list:
        raise ResolutionError(f"species could not be resolved: {species}")
    taxonomy_id = int(id_list[0])
    body = cache.fetch(NCBI_ESUMMARY, params={"db": "taxonomy", "id": id_list[0], "retmode": "json"})
    result = json.loads(body).get("result") or {}
    resolved_name = (result.get(id_list[0]) or {}).get("scientificname") or name
    entry = {"taxonomy_id": taxonomy_id, "resolved_name": resolved_name}
    if taxonomy_cache is not None:
        taxonomy_cache.set(name, entry)
    return entry
