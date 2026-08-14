from __future__ import annotations

import json
import warnings
from pathlib import Path

from protein_engine.config import DEFAULT_TOP_N, PATHS, PRESELECT_LIMIT, RATE_LIMIT_DEFAULT, RETRY_BACKOFF
from protein_engine.ranking import essential, resistance, virulence
from protein_engine.ranking.scorer import preselect, rank
from protein_engine.retrieval import ncbi_client, pdb_client, uniprot_client
from protein_engine.retrieval.api_cache import ApiCache, ApiError
from protein_engine.sequence import exporter, validator
from protein_engine.taxonomy.cache import TaxonomyCache
from protein_engine.taxonomy.resolver import resolve


def load_curated(reference_dir: Path | str) -> dict:
    reference = Path(reference_dir)
    loaders = {
        "essential": ("essential_genes.csv", essential.load_essential_genes),
        "virulence": ("virulence_genes.csv", virulence.load_virulence_genes),
        "resistance": ("resistance_genes.csv", resistance.load_resistance_genes),
    }
    tables = {}
    for name, (filename, loader) in loaders.items():
        path = reference / filename
        if path.exists():
            tables[name] = loader(path)
    return tables


def run(
    species: str,
    *,
    top_n: int = DEFAULT_TOP_N,
    output_dir: Path | str = PATHS.output_dir,
    cache_dir: Path | str = PATHS.cache_dir,
    reference_dir: Path | str = PATHS.reference_dir,
    offline: bool = False,
    transport: object | None = None,
    rate_limit: float | None = None,
) -> dict:
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    cache = ApiCache(
        cache_dir_path / "http.db",
        transport=transport,
        rate_limit=0.0 if offline else (rate_limit if rate_limit is not None else RATE_LIMIT_DEFAULT),
        backoff=RETRY_BACKOFF,
        allow_network=not offline,
    )
    taxonomy_cache = TaxonomyCache(cache_dir_path / "taxonomy.db")

    resolved = resolve(species, cache, taxonomy_cache)
    proteins = uniprot_client.fetch_proteins(resolved["taxonomy_id"], cache)
    source = "uniprot"
    if not proteins:
        proteins = ncbi_client.fetch_proteins(resolved["taxonomy_id"], cache)
        source = "ncbi"
    if not proteins:
        raise RuntimeError(f"no proteins found for species: {species}")

    curated = load_curated(reference_dir)
    shortlist = preselect(proteins, curated=curated, limit=max(PRESELECT_LIMIT, top_n * 5))
    structure_flags = {}
    for p in shortlist:
        accession = p.get("accession")
        if not accession:
            continue
        try:
            structure_flags[accession] = pdb_client.has_structure(accession, cache)
        except ApiError as exc:
            warnings.warn(f"PDB structure lookup failed for {accession}: {exc}", stacklevel=2)
            structure_flags[accession] = False
    ranked = rank(shortlist, structure_flags=structure_flags, curated=curated, top_n=top_n)
    for i, rec in enumerate(ranked, start=1):
        rec["rank"] = i
        rec["has_structure"] = structure_flags.get(rec.get("accession"), False)
    valid, excluded = validator.split_valid(ranked)

    out = exporter.write(
        species,
        resolved["resolved_name"],
        resolved["taxonomy_id"],
        valid,
        excluded,
        output_dir,
        source=source,
    )
    payload = json.loads((out / "proteins.json").read_text())
    payload["output_dir"] = str(out)
    return payload
