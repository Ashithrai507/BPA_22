# Phase 2 — Protein Retrieval & Selection Engine (Design Spec)

**Status:** Draft — pending user review
**Date:** 2026-08-06
**Author:** BPA-22 design session
**Related issues:** #7–#10 (Phase 1 accuracy), Phase 2 scope
**Approach:** A (UniProt-first), selected by user

## 1. Background & Motivation

The Phase 1 CV pipeline (`bacteria_assistant`) predicts a bacterial or fungal
species from a Petri-dish image. Protein folding models (AlphaFold / ESMFold /
OpenFold) cannot accept an image or a species name — they require an amino-acid
sequence. The Protein Engine bridges species prediction to folding by producing
a **validated, ranked amino-acid sequence set** ready for a folding model.

Intended use: **real research / lab use** (user-confirmed). Output must be
curated and ranked by biological relevance, not random.

## 2. Goals & Non-Goals

### Goals
1. Species name → taxonomy ID (NCBI).
2. Retrieve candidate proteins for the species (UniProt primary, NCBI fallback).
3. Rank candidates by research relevance: essential genes, virulence factors,
   antimicrobial-resistance genes, evidence level, structure availability.
4. Validate sequences and emit folding-ready JSON + FASTA.
5. Support any species; run online with a SQLite cache for offline repeatability.
6. Consume via CLI and as a Python library.

### Non-Goals (v1)
- 3D structure prediction (folding). The engine emits folding-ready output;
  a folding adapter is a later extension.
- GUI integration beyond the CLI/library surface (future `bacteria_ui.py` button).
- Fully offline curated reference bundles for all species (future option).

## 3. Locked Design Decisions

| # | Decision | Choice | Rationale |
| - | -------- | ------ | --------- |
| 1 | Primary data source | UniProt REST API | Single rich source: sequence, function, evidence, subcellular location, PDB/AlphaFold cross-refs |
| 2 | Fallback source | NCBI E-utilities | Taxonomy resolution + protein lookup when UniProt returns nothing |
| 3 | Structure info | PDB cross-ref only | `has_structure` flag for scoring; never a peer source |
| 4 | Ranking data | Curated layers (DEG, VFDB, CARD) | Research-grade scoring |
| 5 | Scope | Any species | Not limited to the 10 CV species |
| 6 | Network | Online + SQLite cache | Fast repeat/offline runs; polite rate-limiting |
| 7 | Consumption | CLI + library | `run_protein_pipeline.py` + importable package |
| 8 | Folding | Output contract now, 3D later | v1 emits folding-ready FASTA/JSON; folding adapter is a future extension point |
| 9 | Retrieval narrowing | Reviewed-only first | Server-side filter cuts B. subtilis 4,200 → ~1,500 Swiss-Prot records |
| 10 | New dependency | `requests` only | Everything else is stdlib (`sqlite3`, `json`, `argparse`, `dataclasses`) |

## 4. Architecture

Package under `src/`, importable as `protein_engine`.

```
src/protein_engine/
├── config.py                    # endpoints, weights, paths, rate limits
├── taxonomy/
│   ├── resolver.py              # normalize name → NCBI taxonomy ID
│   └── cache.py                 # species ↔ taxonomy_id SQLite table
├── retrieval/
│   ├── uniprot_client.py        # primary search/fetch
│   ├── ncbi_client.py           # fallback protein + taxonomy lookup
│   ├── pdb_client.py            # structure-availability cross-ref
│   └── api_cache.py             # generic HTTP-response SQLite cache
├── ranking/
│   ├── essential.py             # DEG table loader
│   ├── virulence.py             # VFDB table loader
│   ├── resistance.py            # CARD table loader
│   └── scorer.py                # weighted rank + score_breakdown
├── sequence/
│   ├── validator.py             # alphabet/length/dedup checks
│   ├── fasta_parser.py          # parse incoming FASTA
│   └── exporter.py              # write JSON + FASTA outputs
├── documentation/
│   ├── design.md                # system design
│   └── architecture.md          # technical architecture
└── scripts/run_protein_pipeline.py   # CLI entry
```

## 5. Data Flow

1. **Entry** — `--species "<name>"` or `--image <path>` (runs
   `bacteria_assistant.inference` for the predicted species; requires a model artifact).
2. **Taxonomy resolution** — normalize name → cache lookup → NCBI `esearch`
   (db=`taxonomy`) + `esummary` → `{"taxonomy_id", "resolved_name"}`.
   Unresolved → `ResolutionError`, exit 2.
3. **Retrieval (UniProt primary)** —
   `GET /uniprotkb/search?query=taxonomy_id:<id> AND reviewed:true AND length:[40 TO *]&fields=...&size=500&format=json`,
   paginated via `nextLink`. If reviewed set empty or < 10 → retry `reviewed:false`.
   If still empty → NCBI fallback (`esearch` db=`protein` `txid<id>[Organism]` →
   `efetch` FASTA).
4. **Structure cross-ref** — for ranked shortlist, `GET
   data.rcsb.org/rest/v1/core/uniprot/<acc>` → `has_structure` (cached).
5. **Ranking** — weighted score (§6) + `score_breakdown`, top-N.
6. **Validation** — amino-acid alphabet, length ≥ 20, dedup by accession,
   flag `X/B/Z`; failures → `excluded[]`.
7. **Export** — `output/<slug>/proteins.json` + `sequences.fasta`.

## 6. Scoring Formula

```
score = 0.30×essential + 0.25×virulence + 0.20×resistance
      + 0.15×evidence + 0.10×structure_availability
```

| Term | Source | [0,1] mapping |
| ---- | ------ | ------------- |
| essential | DEG / OGEE table | 1.0 if accession/gene in table, else 0.0 |
| virulence | VFDB table | 1.0 if present, else 0.0 |
| resistance | CARD table | 1.0 if present, else 0.0 |
| evidence | UniProt `reviewed` | 1.0 reviewed, 0.5 unreviewed |
| structure | PDB / AlphaFold cross-ref | 1.0 if structure exists, else 0.0 |

Weights in `config.py` as `RANKING_WEIGHTS`. Curated tables matched by gene name
(primary) then accession; missing table → neutral term + warning. Curated
bundles in `reference_data/*.csv`, gitignored.

## 7. Output Contract

JSON + FASTA written per species. JSON shape:

```json
{
  "species": "Bacillus subtilis",
  "taxonomy_id": 224308,
  "resolved_name": "Bacillus subtilis subsp. subtilis",
  "query_timestamp": "2026-08-06T12:00:00Z",
  "selected_proteins": [
    {
      "rank": 1,
      "accession": "P37476",
      "protein": "Stage 0 sporulation protein A",
      "gene": "spo0A",
      "length": 405,
      "sequence": "MNNKILVADDPLVEL...",
      "score": 0.82,
      "score_breakdown": {"essential": 1.0, "virulence": 0.0, "resistance": 0.0, "evidence": 0.8, "structure": 0.4},
      "evidence": "reviewed",
      "has_structure": false,
      "subcellular_location": "Cytoplasm",
      "function": "Response regulator involved in sporulation initiation"
    }
  ],
  "excluded": []
}
```

FASTA:

```
>P37476|spo0A|Stage 0 sporulation protein A|Bacillus subtilis
MNNKILVADDPLVEL...
```

## 8. Error Handling & Exit Codes

| Stage | Failure | Behaviour |
| ----- | ------- | --------- |
| Taxonomy resolution | Species not found | Clean error, exit 2 |
| Retrieval | Network down, no cache | Clear message, exit 1 |
| Retrieval | Network down, cache hit | Serve from cache (offline) |
| UniProt search | No results | NCBI fallback |
| Ranking | Curated table missing | Neutral term + warning |
| Validation | Invalid sequence | `excluded[]`, pipeline continues |

Exit codes: `0` success, `1` infrastructure/network, `2` user-input/resolution.

## 9. Caching

- `cache/taxonomy.db` — species → taxonomy_id (key: normalized name).
- `cache/http.db` — HTTP responses (key: method + URL + params hash).
- Both SQLite, gitignored, timestamped + `schema_version`.
- `--offline` forces cache-only (clear error if key absent).

## 10. Rate Limiting & Retry

- Token-bucket: 5 req/s UniProt, 3 req/s NCBI (without API key).
- Retry: 0.5 s → 1 s → 2 s, max 3, on 429/5xx; then fallback or cache.

## 11. Testing Strategy

- Unit: validator, FASTA parser, exporter, scorer, cache (no network).
- HTTP: mock transport (`responses`), never live APIs in CI.
- Fixtures: small CSV curated tables.
- E2E: stubbed UniProt response for `Bacillus subtilis` vs JSON contract.
- Offline/exit-code tests for error paths.

## 12. Milestones

| M | Scope | Deliverable |
| - | ----- | ----------- |
| M1 | Taxonomy resolver + cache | taxonomy_id from species name, cached |
| M2 | UniProt/NCBI retrieval + cache | candidate protein records, offline repeatable |
| M3 | Ranking engine | top-N with score + rationale |
| M4 | Validation + export | validated JSON + FASTA |
| M5 | Folding integration | documented contract; (v2) folding adapter → 3D |

## 13. Open Questions / Risks

- DEG/VFDB/CARD coverage per species: matched by gene name to maximize portability;
  neutral-term fallback when absent. Validation needed on real species.
- UniProt taxonomy ID correctness depends on NCBI resolution; requires the
  `esummary` canonical-name check.
- `requests` introduces the first non-stdlib runtime dep for the engine; acceptable.
- Live-API smoke test (not in CI) recommended before M2 sign-off.
