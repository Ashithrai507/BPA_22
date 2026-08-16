from __future__ import annotations

import json

from protein_engine.config import PDB_UNIPROT
from protein_engine.retrieval.api_cache import ApiCache, ApiError


def has_structure(accession: str, cache: ApiCache) -> bool:
    url = f"{PDB_UNIPROT}/{accession}"
    try:
        body = cache.fetch(url)
    except ApiError as exc:
        if exc.status_code == 404:
            cache.set(url, None, "null")
            return False
        raise
    return bool(json.loads(body))
