from __future__ import annotations

import json

from protein_engine.config import PAGE_SIZE, REVIEWED_MIN_COUNT, UNIPROT_FIELDS, UNIPROT_SEARCH
from protein_engine.retrieval.api_cache import ApiCache


def _search(taxonomy_id: int, cache: ApiCache, reviewed: bool) -> list[dict]:
    query = f"taxonomy_id:{taxonomy_id} AND reviewed:{str(reviewed).lower()} AND length:[40 TO *]"
    params = {"query": query, "fields": UNIPROT_FIELDS, "format": "json", "size": PAGE_SIZE}
    results: list[dict] = []
    url = UNIPROT_SEARCH
    while True:
        body = cache.fetch(url, params=params)
        data = json.loads(body)
        results.extend(data.get("results") or [])
        next_link = data.get("nextLink")
        if not next_link:
            break
        url = next_link
        params = None
    return results


def fetch_proteins(taxonomy_id: int, cache: ApiCache, reviewed_min_count: int = REVIEWED_MIN_COUNT) -> list[dict]:
    reviewed_results = _search(taxonomy_id, cache, reviewed=True)
    if len(reviewed_results) >= reviewed_min_count:
        return [normalize(r, reviewed=True) for r in reviewed_results]
    unreviewed_results = _search(taxonomy_id, cache, reviewed=False)
    if unreviewed_results:
        return [normalize(r, reviewed=False) for r in unreviewed_results]
    return [normalize(r, reviewed=True) for r in reviewed_results]


def normalize(result: dict, reviewed: bool) -> dict:
    seq = result.get("sequence") or {}
    desc = result.get("proteinDescription") or {}
    protein_name = ((desc.get("recommendedName") or {}).get("fullName") or {}).get("value")
    if not protein_name:
        submissions = desc.get("submissionNames") or []
        if submissions:
            protein_name = ((submissions[0].get("fullName")) or {}).get("value")
    gene = None
    for entry in result.get("genes") or []:
        value = (entry.get("geneName") or {}).get("value")
        if value:
            gene = value
            break
    subcellular = None
    for loc in result.get("subcellularLocations") or []:
        subcellular = (loc.get("location") or {}).get("value")
        if subcellular:
            break
    function = None
    for comment in result.get("comments") or []:
        if comment.get("commentType") == "FUNCTION" and comment.get("texts"):
            function = comment["texts"][0].get("value")
            break
    pdb = [
        x.get("id")
        for x in (result.get("uniProtKBCrossReferences") or [])
        if x.get("database") == "PDB" and x.get("id")
    ]
    return {
        "accession": result.get("primaryAccession"),
        "protein_name": protein_name,
        "gene": gene,
        "length": seq.get("length", len(seq.get("value", ""))),
        "sequence": seq.get("value", ""),
        "reviewed": bool(reviewed),
        "evidence": "reviewed" if reviewed else "unreviewed",
        "subcellular_location": subcellular,
        "function": function,
        "pdb_structures": pdb,
        "keywords": [k.get("name") for k in result.get("keywords") or [] if k.get("name")],
    }
