# Protein Engine — System Design

> Phase 2 of the BPA-22 project. Bridges the computer-vision species prediction
> to a validated amino-acid sequence ready for protein folding.

## 1. Purpose

The CV pipeline (`bacteria_assistant`) identifies a bacterial or fungal species
from a Petri-dish image. Protein folding models (AlphaFold / ESMFold / OpenFold)
cannot operate on an image or a species name — they require an amino-acid
sequence. This engine fills that gap:

```
Petri Dish Image ──► CV Pipeline ──► Species Prediction ──► Protein Engine
    ──► Taxonomy ID ──► Protein Retrieval ──► Ranking ──► Validated Sequence
    ──► (folding adapter, future)
```

The final deliverable is a validated amino-acid sequence (or a ranked set)
that can be passed directly to a folding model.

## 2. Design Decisions (locked in)

| Decision | Choice | Rationale |
| -------- | ------ | --------- |
| Primary data source | **UniProt REST API** | Single rich source: accession, name, gene, function, length, sequence, evidence, subcellular location, cross-refs, PDB links |
| Fallback source | **NCBI E-utilities** | Taxonomy resolution + protein lookup when UniProt returns nothing |
| Structure info | **PDB cross-ref only** | `has_structure` flag for scoring; never a peer source |
| Ranking data | **Curated layers** (DEG, VFDB, CARD) | Research-grade scoring via essential / virulence / resistance tables |
| Scope | **Any species** | Not limited to the 10 CV species |
| Network | **Online + SQLite cache** | Repeat runs offline and fast; polite rate-limiting |
| Consumption | **CLI + library** | `run_protein_pipeline.py` + importable package |
| Folding | **Output contract now, 3D later** | v1 emits folding-ready FASTA/JSON; a folding adapter is a future extension point |
| Biopython | **Not used in v1** | UniProt REST covers everything; FASTA parsing is trivial |
| New dependency | **`requests` only** | Everything else is stdlib (`sqlite3`, `json`, `urllib`) |

## 3. Output Contract

The pipeline always produces structured JSON plus a FASTA file.

### 3.1 JSON

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
      "score_breakdown": {
        "essential": 1.0,
        "virulence": 0.0,
        "resistance": 0.0,
        "evidence": 0.8,
        "structure": 0.4
      },
      "evidence": "reviewed",
      "has_structure": false,
      "subcellular_location": "Cytoplasm",
      "function": "Response regulator involved in sporulation initiation"
    }
  ]
}
```

### 3.2 FASTA

```fasta
>P37476|spo0A|Stage 0 sporulation protein A|Bacillus subtilis
MNNKILVADDPLVEL...
```

### 3.3 Folding-ready contract (M5)

The JSON + FASTA are the agreed interface a folding adapter consumes. A future
folding adapter (e.g. ESMFold/ColabFold) reads `sequences.fasta`, runs folding,
and writes structure files (`*.pdb`) into the same `output/` tree — no change to
the engine's internals is required.

## 4. Retrieval Deep-Dive

### 4.1 Roles

| Source | Role | Why |
| ------ | ---- | --- |
| **UniProt REST** | Primary | One call returns sequence + metadata + PDB cross-refs |
| **NCBI E-utilities** | Fallback | Only if UniProt yields nothing for the taxid |
| **PDB (RCSB REST)** | Cross-ref flag | Sets `has_structure` for scoring; never a peer source |

### 4.2 UniProt (primary)

Search the species proteome, filtered server-side, paginated:

```
GET https://rest.uniprot.org/uniprotkb/search
  ?query=taxonomy_id:224308 AND reviewed:true AND length:[40 TO *]
  &fields=accession,reviewed,protein_name,gene_names,length,sequence,
          cc_function,subcellular_location,xref_pdb,xref_alphafold
  &size=500
  &format=json
```

- **Reviewed-only first (locked in).** Start with `reviewed:true`. If the result
  set is empty or < 10 records, retry with `reviewed:false`. Server-side filtering
  avoids downloading the full proteome (B. subtilis ≈ 4,200 proteins → ~1,500
  Swiss-Prot records).
- **Pagination.** UniProt returns a `nextLink` cursor; walk it until empty.
- **One call carries the sequence** (in `sequence.value`) and the PDB cross-refs,
  so ranking data costs no extra round-trips.

Normalized record shape:

```json
{
  "accession": "P37476",
  "protein_name": "Stage 0 sporulation protein A",
  "gene": "spo0A",
  "length": 405,
  "sequence": "MNNKILVADDPLVEL...",
  "reviewed": true,
  "subcellular_location": "Cytoplasm",
  "pdb_structures": ["1FSE"],
  "keywords": ["Sporulation", "Phosphoprotein"]
}
```

### 4.3 Rate limiting, retry, cache

- **Rate limit:** token-bucket, 5 req/s for UniProt (their guidance is ≤10),
  3 req/s for NCBI without an API key.
- **Retry:** exponential backoff (0.5 s → 1 s → 2 s, max 3) on 429/5xx, then
  fallback or cache.
- **Cache:** every HTTP response stored in `cache/http.db` keyed by
  `(method, url, params-hash)`. `--offline` serves cache-only.

### 4.4 NCBI fallback (only if UniProt yields nothing)

```
esearch.fcgi?db=protein&term=txid224308[Organism]&retmode=json   → protein IDs
efetch.fcgi?db=protein&id=<ids>&rettype=fasta&retmode=text       → sequences
```

Also used for taxonomy resolution (db=`taxonomy`) when a species name is unresolved.

### 4.5 PDB cross-ref

```
GET https://data.rcsb.org/rest/v1/core/uniprot/P37476   → list of polymer entities
```

`has_structure = true` if any entity exists. Cached per accession, only queried
for the ranked shortlist (top-N).

### 4.6 Payload estimate

~1,500 reviewed records × ~1 KB normalized ≈ 1.5 MB across ~3 pages. Fast enough
for interactive use; cached for repeat/offline runs.

## 5. Error Handling

Every stage degrades gracefully and never produces silently-empty output:

| Stage | Failure | Behaviour |
| ----- | ------- | --------- |
| Taxonomy resolution | Species not found | Clean error, exit code 2 |
| Retrieval | Network down, no cache | Clear message, exit code 1 |
| Retrieval | Network down, cache hit | Serve from cache (offline mode) |
| UniProt search | No results | Fall back to NCBI protein search |
| Ranking | Curated table missing | Score that term as neutral (0.0) + warning |
| Sequence validation | Empty / too short / invalid alphabet | Candidate excluded from output + listed in `excluded[]` |

Recommended exit codes:
- `0` success
- `1` infrastructure / network error
- `2` user-input / resolution error

## 6. Dependencies

- Add `requests` to `requirements.txt`.
- `sqlite3`, `json`, `argparse`, `dataclasses` — standard library.
- No Biopython, no ML-lib additions for v1.

## 7. Testing Strategy

- **Unit tests** for validator, FASTA parser, exporter, scorer, cache — no network.
- **HTTP tests** using a mock transport (`responses` or a fake session) — never
  hit live UniProt/NCBI in CI.
- **Fixture curated tables** — small CSV fixtures for essential/virulence/resistance.
- **End-to-end test** — stubbed UniProt response for `Bacillus subtilis` verified
  against the JSON contract.
- Offline/exit-code tests for the error paths above.

## 8. Milestones

| Milestone | Scope | Deliverable |
| --------- | ----- | ----------- |
| M1 | Species normalization + taxonomy resolver | `taxonomy_id` from a species name, cached |
| M2 | UniProt/NCBI retrieval + SQLite cache | candidate protein records, offline repeatable |
| M3 | Ranking engine | top-N proteins with score + rationale |
| M4 | Sequence validation + FASTA export | validated JSON + FASTA files |
| M5 | Folding integration | documented contract; (v2) folding adapter producing 3D structures |
