# Protein Engine — Architecture

Detailed technical architecture for the protein retrieval & selection engine.

## 1. Layer Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                         CLI / Library                              │
│                scripts/run_protein_pipeline.py                     │
│              (--species | --image, --top-n, --offline)             │
└───────────────────────────────┬────────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────────┐
│                        Orchestration                               │
│        taxonomy → retrieval → ranking → sequence → exporter        │
└───────────────────────────────┬────────────────────────────────────┘
                                │
        ┌───────────────┬───────┴───────┬───────────────┐
        ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   taxonomy   │ │  retrieval   │ │   ranking    │ │   sequence   │
│ resolver+    │ │ uniprot      │ │ essential    │ │ validator    │
│ cache        │ │ ncbi (fb)    │ │ virulence    │ │ fasta_parser │
│              │ │ pdb (xref)   │ │ resistance   │ │ exporter     │
│              │ │ api_cache    │ │ scorer       │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │                │
       ▼                ▼                ▼                ▼
┌────────────────────────────────────────────────────────────────────┐
│                      External / Persistent                         │
│   NCBI E-utilities │ UniProt REST │ PDB │ SQLite cache │           │
│   reference_data/ (DEG/VFDB/CARD) │ output/                        │
└────────────────────────────────────────────────────────────────────┘
```

## 2. Package Structure

```
src/protein_engine/
├── __init__.py
├── config.py                    # endpoints, weights, paths, rate limits
├── pipeline.py                  # run(): orchestrates taxonomy→retrieval→ranking→export
├── taxonomy/
│   ├── __init__.py
│   ├── resolver.py              # normalize name → NCBI taxonomy ID
│   └── cache.py                 # species ↔ taxonomy_id SQLite table
├── retrieval/
│   ├── __init__.py
│   ├── uniprot_client.py        # primary search/fetch
│   ├── ncbi_client.py           # fallback protein + taxonomy lookup
│   ├── pdb_client.py            # structure-availability cross-ref
│   └── api_cache.py             # generic HTTP-response SQLite cache
├── ranking/
│   ├── __init__.py
│   ├── essential.py             # DEG table loader
│   ├── virulence.py             # VFDB table loader
│   ├── resistance.py            # CARD table loader
│   └── scorer.py                # weighted rank + score_breakdown
├── sequence/
│   ├── __init__.py
│   ├── validator.py             # alphabet/length/dedup checks
│   ├── fasta_parser.py          # parse incoming FASTA
│   └── exporter.py              # write JSON + FASTA outputs
├── documentation/
│   ├── design.md                # system design (this engine)
│   └── architecture.md          # this document
└── (runtime, gitignored)
    ├── models/                  # reserved: cached/fitted ranker state
    └── output/                  # per-species results (JSON + FASTA)
```

## 3. Data Flow (detailed)

1. **Entry** — `run_protein_pipeline.py` accepts `--species "<name>"` or
   `--image <path>` (if `--image`, run `bacteria_assistant.inference` to get the
   predicted species first; requires a trained model artifact).
2. **Taxonomy resolution** — `resolver.resolve(species)`:
   - Normalize the scientific name (strip casing, collapse whitespace).
   - Check `cache.py` for a stored `(species → taxonomy_id)`.
   - Cache miss → NCBI E-utilities `esearch` (db=`taxonomy`) then `esummary` for
     the canonical ID and resolved name. Store in cache.
   - No result → raise `ResolutionError` (exit 2).
3. **Retrieval** — `uniprot_client.fetch_proteins(taxonomy_id)`:
   - Query UniProt `search` by `taxonomy_id:{id}`, filtered server-side:
     `taxonomy_id:<id> AND reviewed:true AND length:[40 TO *]`, paginated
     (`size=500`, `nextLink` cursor).
   - Requested fields: `accession, protein_name, gene_names, length, sequence,
     reviewed, subcellular_location, keywords, cc_function, xref_pdb, xref_alphafold`.
   - **Reviewed-only first (locked in):** if the reviewed result set is empty or
     < 10 records, retry with `reviewed:false`.
   - Polite token-bucket rate limit (3 req/s shared across endpoints, see
     `RATE_LIMIT_DEFAULT`), retry with
     exponential backoff, cache each response in `api_cache.py` (keyed by query + page).
   - Empty result → fallback `ncbi_client.fetch_proteins(taxonomy_id)`
     (`esearch` db=`protein` by `txid<id>[Organism]` → `efetch` FASTA).
4. **Structure cross-ref** — for the ranked shortlist, `pdb_client.has_structure(accession)`
   sets the `has_structure` flag (cached) via
   `https://data.rcsb.org/rest/v1/core/uniprot/<accession>`.
5. **Ranking** — `scorer.rank(proteins, taxonomy_id)`:
   - Load curated tables (`essential/virulence/resistance` loaders) mapped by
     UniProt accession (fallback: gene name).
   - Compute the weighted score (§4) + `score_breakdown`.
   - Sort descending, return `top_n`.
6. **Validation** — `validator.validate(protein)`:
   - 20-letter amino-acid alphabet check.
   - Minimum length guard (default 20 aa).
   - Duplicate-accession removal; unknown residues (`X/B/Z`) flagged, not silently
     dropped from the record.
   - Candidates failing validation are recorded in `excluded[]`.
7. **Export** — `exporter.write(species, proteins, excluded)`:
   - Writes `output/<slug>/proteins.json` and `output/<slug>/sequences.fasta`.
   - JSON follows the §3 contract in `design.md`.

## 4. Scoring

```
score = 0.30×essential + 0.25×virulence + 0.20×resistance
      + 0.15×evidence + 0.10×structure_availability
```

| Term | Source | [0,1] mapping |
| ---- | ------ | ------------- |
| essential | DEG / OGEE table | 1.0 if accession/gene in table, else 0.0 |
| virulence | VFDB table | 1.0 if present, else 0.0 |
| resistance | CARD table | 1.0 if present, else 0.0 |
| evidence | UniProt `reviewed` | 1.0 reviewed (Swiss-Prot), 0.5 unreviewed |
| structure | PDB / AlphaFold cross-ref | 1.0 if structure exists, else 0.0 |

Weights live in `config.py` as `RANKING_WEIGHTS` so they can be tuned without
code changes. If a curated table is missing for a given taxonomy, its term
defaults to 0.0 and a warning is emitted.

## 5. Caching Design

Two SQLite databases (stdlib `sqlite3`), both gitignored:

| Database | Purpose | Key |
| -------- | ------- | --- |
| `cache/taxonomy.db` | species → taxonomy_id | normalized species name |
| `cache/http.db` | arbitrary HTTP responses | method + URL + query hash |

- Every cache row stores a timestamp + `schema_version`.
- `--offline` flag forces cache-only: resolution/retrieval fail with a clear
  message if the key is absent, instead of hitting the network.
- Caches are append-only and safe to delete.

## 6. Interfaces (public functions)

```python
# taxonomy/resolver.py
def resolve(species: str, cache: ApiCache, taxonomy_cache: TaxonomyCache | None = None) -> dict: ...
    # {"taxonomy_id": int, "resolved_name": str}

# retrieval/uniprot_client.py
def fetch_proteins(taxonomy_id: int, cache: ApiCache, reviewed_min_count: int = 10) -> list[dict]: ...

# retrieval/ncbi_client.py
def fetch_proteins(taxonomy_id: int, cache: ApiCache, retmax: int = 200) -> list[dict]: ...  # fallback

# retrieval/pdb_client.py
def has_structure(accession: str, cache: ApiCache) -> bool: ...

# ranking/_common.py              # shared CSV loader
def load_curated_table(path: Path, name: str, gene_columns, accession_columns) -> CuratedTable: ...
class CuratedTable: ...            # .matches(protein) -> bool

# ranking/essential.py
def load_essential_genes(path: Path) -> CuratedTable: ...

# ranking/virulence.py
def load_virulence_genes(path: Path) -> CuratedTable: ...

# ranking/resistance.py
def load_resistance_genes(path: Path) -> CuratedTable: ...

# ranking/scorer.py
def score_terms(protein: dict, curated: dict | None = None, has_structure: bool = False,
                weights: dict | None = None) -> dict: ...
def preselect(proteins: list[dict], curated: dict | None = None, limit: int = PRESELECT_LIMIT) -> list[dict]: ...
def rank(proteins: list[dict], structure_flags: dict | None = None, curated: dict | None = None,
         weights: dict | None = None, top_n: int = DEFAULT_TOP_N) -> list[dict]: ...

# sequence/validator.py
def validate(protein: dict, min_length: int = MIN_SEQUENCE_LENGTH) -> dict: ...
    # {"ok": bool, "protein": dict, "issues": list[str]}
def split_valid(proteins: list[dict], min_length: int = MIN_SEQUENCE_LENGTH) -> tuple[list[dict], list[dict]]: ...

# sequence/exporter.py
def write(species: str, resolved_name: str, taxonomy_id: int, proteins: list[dict],
          excluded: list[dict], output_dir: Path | str, source: str = "uniprot") -> Path: ...

# pipeline.py                    # orchestrator
def run(species: str, *, top_n: int = DEFAULT_TOP_N, output_dir: Path | str = PATHS.output_dir,
        cache_dir: Path | str = PATHS.cache_dir, reference_dir: Path | str = PATHS.reference_dir,
        offline: bool = False, transport: object | None = None, rate_limit: float | None = None) -> dict: ...
def load_curated(reference_dir: Path | str) -> dict: ...

# scripts/run_protein_pipeline.py
def main(argv: list[str] | None = None) -> int: ...
```

## 7. Configuration

`protein_engine/config.py` holds:

- API base URLs: UniProt `https://rest.uniprot.org/`, NCBI
  `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`, PDB
  `https://data.rcsb.org/rest/v1/`.
- Rate limits (requests/sec): 3 shared across all endpoints (token-bucket,
  `RATE_LIMIT_DEFAULT`); retry/backoff 0.5 s → 1 s → 2 s, max 3.
- `RANKING_WEIGHTS`.
- Cache/output directory defaults (overridable via `.env`:
  `PROTEIN_CACHE_DIR`, `PROTEIN_OUTPUT_DIR`, `PROTEIN_REFERENCE_DIR`).
- Validation constants: `MIN_SEQUENCE_LENGTH = 20`.

## 8. Failure Modes & Fallbacks

1. **UniProt down →** retry/backoff; then NCBI fallback; then cache.
2. **Taxonomy un-resolvable →** exit 2 with the offending name.
3. **Curated table unavailable →** neutral score term + warning.
4. **No proteins found →** exit 1 with a clear message (never empty JSON).
5. **Invalid sequence →** candidate moved to `excluded[]`, pipeline continues.

## 9. Future Extensions

- **Folding adapter (v2):** ESMFold/ColabFold reads `sequences.fasta`, writes
  `*.pdb` into `output/` — a new adapter module, no engine changes.
- **Per-species curated bundles:** optional `reference_data/` snapshots for the 10
  CV species (hybrid offline mode).
- **GUI integration:** a button in `bacteria_ui.py` invoking the engine on the
  last prediction.
