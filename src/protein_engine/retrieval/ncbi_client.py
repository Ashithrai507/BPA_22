from __future__ import annotations

import json

from protein_engine.config import NCBI_EFETCH, NCBI_ESEARCH
from protein_engine.retrieval.api_cache import ApiCache
from protein_engine.sequence.fasta_parser import parse


def fetch_proteins(taxonomy_id: int, cache: ApiCache, retmax: int = 200) -> list[dict]:
    body = cache.fetch(
        NCBI_ESEARCH,
        params={"db": "protein", "term": f"txid{taxonomy_id}[Organism]", "retmode": "json", "retmax": retmax},
    )
    id_list = ((json.loads(body).get("esearchresult") or {}).get("idlist")) or []
    if not id_list:
        return []
    text = cache.fetch(
        NCBI_EFETCH,
        params={"db": "protein", "id": ",".join(id_list), "rettype": "fasta", "retmode": "text"},
    )
    return [_normalize(record) for record in parse(text)]


def _normalize(record: dict) -> dict:
    return {
        "accession": record["accession"],
        "protein_name": record["header"],
        "gene": None,
        "length": len(record["sequence"]),
        "sequence": record["sequence"],
        "reviewed": False,
        "evidence": "unreviewed",
        "subcellular_location": None,
        "function": None,
        "pdb_structures": [],
        "keywords": [],
    }
