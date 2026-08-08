# Protein Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 2 Protein Retrieval & Selection Engine — a CLI + library that turns a species name (or CV-predicted species) into a ranked, validated, folding-ready FASTA/JSON protein set via UniProt (primary), NCBI (fallback), and PDB (structure cross-ref).

**Architecture:** Layered pipeline under `src/protein_engine/`. `taxonomy` resolves species → NCBI taxonomy ID; `retrieval` fetches candidate proteins (UniProt reviewed-first, paginated; NCBI fallback) with a shared rate-limited, retrying, SQLite-backed HTTP cache; `ranking` scores candidates by essential/virulence/resistance curated tables + evidence + structure availability; `sequence` validates and exports JSON + FASTA. A `pipeline.py` orchestrator ties the stages; `scripts/run_protein_pipeline.py` is the thin CLI.

**Tech Stack:** Python 3.11+, stdlib (`sqlite3`, `csv`, `json`, `argparse`, `dataclasses`, `hashlib`, `time`), `requests` (only new runtime dep), pytest, ruff.

## Global Constraints

- Python `>=3.11` (per `pyproject.toml`), ruff line-length `120`, target py311.
- Only new runtime dependency: `requests` (pin `requests==2.32.3` in `requirements.txt`). Tests use an in-repo `FakeTransport`, NOT `responses`.
- Test framework: pytest. `pyproject.toml` already sets `pythonpath = ["src"]` and `testpaths = ["tests"]`, so tests import `from protein_engine...` directly.
- All network access in tests must go through `tests/protein_engine/conftest.py`'s `FakeTransport` — never hit live UniProt/NCBI/PDB.
- Every public function returns plain dicts/lists; no dataclass leaks across module boundaries.
- Commit after every green task using conventional-commit messages (`feat:`, `fix:`, `test:`).
- Do not modify `src/bacteria_assistant/` except reading `bacteria_assistant.config`/`bacteria_assistant.inference` from the CLI `--image` branch.
- Spec reference: `docs/superpowers/specs/2026-08-06-protein-engine-design.md`. Docs to keep in sync: `src/protein_engine/documentation/{design,architecture}.md`.

---

### Task 1: Config, deps, and ruff isort

**Files:**
- Create: `src/protein_engine/config.py`
- Modify: `requirements.txt`, `pyproject.toml`

**Interfaces:**
- Consumes: nothing.
- Produces: `PATHS` (with `cache_dir`, `output_dir`, `reference_dir`), endpoint constants `UNIPROT_SEARCH`, `NCBI_ESEARCH`, `NCBI_ESUMMARY`, `NCBI_EFETCH`, `PDB_UNIPROT`, `UNIPROT_FIELDS`, `PAGE_SIZE`, `REVIEWED_MIN_COUNT`, `MIN_SEQUENCE_LENGTH`, `DEFAULT_TOP_N`, `PRESELECT_LIMIT`, `RANKING_WEIGHTS`, `RATE_LIMIT_DEFAULT`, `RETRY_BACKOFF`.

- [ ] **Step 1: Write the failing test**

Create `tests/protein_engine/test_config.py`:

```python
from __future__ import annotations

from protein_engine.config import (
    DEFAULT_TOP_N,
    MIN_SEQUENCE_LENGTH,
    PAGE_SIZE,
    PRESELECT_LIMIT,
    RANKING_WEIGHTS,
    RATE_LIMIT_DEFAULT,
    REVIEWED_MIN_COUNT,
    RETRY_BACKOFF,
    UNIPROT_FIELDS,
    UNIPROT_SEARCH,
)


def test_config_constants() -> None:
    assert UNIPROT_SEARCH == "https://rest.uniprot.org/uniprotkb/search"
    assert REVIEWED_MIN_COUNT == 10
    assert MIN_SEQUENCE_LENGTH == 20
    assert PAGE_SIZE == 500
    assert DEFAULT_TOP_N == 10
    assert PRESELECT_LIMIT == 50
    assert RATE_LIMIT_DEFAULT == 3.0
    assert RETRY_BACKOFF == (0.5, 1.0, 2.0)
    assert set(RANKING_WEIGHTS) == {"essential", "virulence", "resistance", "evidence", "structure"}
    assert "sequence" in UNIPROT_FIELDS
```

Also create `tests/protein_engine/conftest.py` placeholder — NO, defer to Task 3. Do NOT create `tests/protein_engine/__init__.py`: pytest auto-inserts non-package test dirs into `sys.path`, which is what makes `from conftest import ...` work in later tasks.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/protein_engine/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'protein_engine.config'`

- [ ] **Step 3: Write minimal implementation**

Create `src/protein_engine/config.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True)
class Paths:
    cache_dir: Path = _env_path("PROTEIN_CACHE_DIR", PROJECT_ROOT / "cache")
    output_dir: Path = _env_path("PROTEIN_OUTPUT_DIR", PROJECT_ROOT / "output")
    reference_dir: Path = _env_path("PROTEIN_REFERENCE_DIR", PROJECT_ROOT / "reference_data")


PATHS = Paths()

UNIPROT_BASE = "https://rest.uniprot.org"
UNIPROT_SEARCH = f"{UNIPROT_BASE}/uniprotkb/search"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_ESEARCH = f"{NCBI_BASE}/esearch.fcgi"
NCBI_ESUMMARY = f"{NCBI_BASE}/esummary.fcgi"
NCBI_EFETCH = f"{NCBI_BASE}/efetch.fcgi"
PDB_UNIPROT = "https://data.rcsb.org/rest/v1/core/uniprot"

UNIPROT_FIELDS = (
    "accession,reviewed,protein_name,gene_names,length,sequence,"
    "cc_function,subcellular_location,xref_pdb,xref_alphafold"
)
REVIEWED_MIN_COUNT = 10
MIN_SEQUENCE_LENGTH = 20
PAGE_SIZE = 500
DEFAULT_TOP_N = 10
PRESELECT_LIMIT = 50

RANKING_WEIGHTS = {
    "essential": 0.30,
    "virulence": 0.25,
    "resistance": 0.20,
    "evidence": 0.15,
    "structure": 0.10,
}
RATE_LIMIT_DEFAULT = 3.0
RETRY_BACKOFF = (0.5, 1.0, 2.0)
```

Modify `requirements.txt` — append:

```
requests==2.32.3
```

Modify `pyproject.toml` — add `"protein_engine"` to `known-first-party`:

```toml
[tool.ruff.lint.isort]
known-first-party = ["bacteria_assistant", "protein_engine"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/protein_engine/test_config.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pyproject.toml src/protein_engine/config.py tests/protein_engine/test_config.py
git commit -m "feat: add protein engine config, deps, and ruff isort"
```

---

### Task 2: TaxonomyCache (SQLite species ↔ taxonomy_id)

**Files:**
- Create: `src/protein_engine/taxonomy/cache.py`
- Test: `tests/protein_engine/test_taxonomy_cache.py`

**Interfaces:**
- Consumes: nothing (stdlib `sqlite3`, `datetime`).
- Produces:
  - `TaxonomyCache(path: Path)` — `get(species: str) -> dict | None` returning `{"taxonomy_id": int, "resolved_name": str}`; `set(species: str, entry: dict) -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/protein_engine/test_taxonomy_cache.py`:

```python
from __future__ import annotations

from protein_engine.taxonomy.cache import TaxonomyCache


def test_round_trip(tmp_path) -> None:
    cache = TaxonomyCache(tmp_path / "taxonomy.db")
    assert cache.get("bacillus subtilis") is None
    cache.set("bacillus subtilis", {"taxonomy_id": 224308, "resolved_name": "Bacillus subtilis subsp. subtilis"})
    assert cache.get("bacillus subtilis") == {"taxonomy_id": 224308, "resolved_name": "Bacillus subtilis subsp. subtilis"}


def test_set_overwrites(tmp_path) -> None:
    cache = TaxonomyCache(tmp_path / "taxonomy.db")
    cache.set("ecoli", {"taxonomy_id": 1, "resolved_name": "x"})
    cache.set("ecoli", {"taxonomy_id": 2, "resolved_name": "y"})
    assert cache.get("ecoli") == {"taxonomy_id": 2, "resolved_name": "y"}


def test_persists_across_instances(tmp_path) -> None:
    path = tmp_path / "taxonomy.db"
    TaxonomyCache(path).set("ecoli", {"taxonomy_id": 562, "resolved_name": "Escherichia coli"})
    assert TaxonomyCache(path).get("ecoli") == {"taxonomy_id": 562, "resolved_name": "Escherichia coli"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/protein_engine/test_taxonomy_cache.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/protein_engine/taxonomy/cache.py`:

```python
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class TaxonomyCache:
    def __init__(self, path: Path) -> None:
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS species_cache ("
            "species TEXT PRIMARY KEY, taxonomy_id INTEGER NOT NULL, "
            "resolved_name TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        self._db.commit()

    def get(self, species: str) -> dict | None:
        row = self._db.execute(
            "SELECT taxonomy_id, resolved_name FROM species_cache WHERE species = ?", (species,)
        ).fetchone()
        return {"taxonomy_id": row[0], "resolved_name": row[1]} if row else None

    def set(self, species: str, entry: dict) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO species_cache (species, taxonomy_id, resolved_name, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (species, entry["taxonomy_id"], entry["resolved_name"], datetime.now(timezone.utc).isoformat()),
        )
        self._db.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/protein_engine/test_taxonomy_cache.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/protein_engine/taxonomy/cache.py tests/protein_engine/test_taxonomy_cache.py
git commit -m "feat: add TaxonomyCache sqlite store"
```

---

### Task 3: ApiCache (shared HTTP cache, rate limit, retry) + FakeTransport test double

**Files:**
- Create: `src/protein_engine/retrieval/api_cache.py`
- Create: `tests/protein_engine/conftest.py`
- Test: `tests/protein_engine/test_api_cache.py`

**Interfaces:**
- Consumes: `config` (`RATE_LIMIT_DEFAULT`, `RETRY_BACKOFF`).
- Produces:
  - `ApiError(status_code: int, url: str)` — `RuntimeError` subclass, attribute `status_code`.
  - `CacheMissError` — `RuntimeError` subclass.
  - `ApiCache(path: Path, transport: object | None = None, rate_limit: float = RATE_LIMIT_DEFAULT, backoff: Sequence[float] = RETRY_BACKOFF)` with:
    - `fetch(url: str, params: dict | None = None, headers: dict | None = None, allow_network: bool = True) -> str`
    - `get(url: str, params: dict | None = None) -> str | None`
    - `set(url: str, params: dict | None, body: str) -> None`
  - `transport` must expose `get(url: str, params: dict | None = None, headers: dict | None = None) -> object` returning an object with `.text: str` and `.status_code: int`. Default transport is the `requests` module.

- [ ] **Step 1: Write the failing tests + conftest**

Create `tests/protein_engine/conftest.py`:

```python
from __future__ import annotations

import json

import pytest


class FakeResponse:
    def __init__(self, text: str = "", status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


class FakeTransport:
    def __init__(self, responses: list[FakeResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def get(self, url: str, params: dict | None = None, headers: dict | None = None) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "headers": headers})
        if not self.responses:
            raise AssertionError(f"unexpected request: {url} {params}")
        return self.responses.pop(0)


def json_response(payload: object, status_code: int = 200) -> FakeResponse:
    return FakeResponse(json.dumps(payload), status_code=status_code)


@pytest.fixture
def fake_transport() -> FakeTransport:
    return FakeTransport()
```

Create `tests/protein_engine/test_api_cache.py`:

```python
from __future__ import annotations

import pytest

from protein_engine.retrieval.api_cache import ApiCache, ApiError, CacheMissError

from conftest import FakeResponse, FakeTransport, json_response


def _cache(path, transport, **kwargs):
    return ApiCache(path, transport=transport, rate_limit=0.0, backoff=[], **kwargs)


def test_fetch_caches_and_serves_from_cache(tmp_path) -> None:
    transport = FakeTransport([json_response({"a": 1})])
    cache = _cache(tmp_path / "http.db", transport)
    assert cache.fetch("http://x", params={"q": "1"}) == '{"a": 1}'
    assert cache.fetch("http://x", params={"q": "1"}) == '{"a": 1}'
    assert len(transport.calls) == 1


def test_fetch_retries_then_succeeds(tmp_path) -> None:
    transport = FakeTransport([FakeResponse("", status_code=503), FakeResponse("ok")])
    cache = ApiCache(tmp_path / "http.db", transport=transport, rate_limit=0.0, backoff=[0.0, 0.0])
    assert cache.fetch("http://x") == "ok"
    assert len(transport.calls) == 2


def test_fetch_raises_api_error_on_non_retryable_status(tmp_path) -> None:
    transport = FakeTransport([FakeResponse("", status_code=404)])
    cache = _cache(tmp_path / "http.db", transport)
    with pytest.raises(ApiError) as excinfo:
        cache.fetch("http://x")
    assert excinfo.value.status_code == 404


def test_fetch_offline_uses_cache_else_raises(tmp_path) -> None:
    transport = FakeTransport([json_response({"a": 1})])
    cache = _cache(tmp_path / "http.db", transport)
    cache.fetch("http://x")
    assert cache.fetch("http://x", allow_network=False) == '{"a": 1}'
    with pytest.raises(CacheMissError):
        cache.fetch("http://other", allow_network=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/protein_engine/test_api_cache.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/protein_engine/retrieval/api_cache.py`:

```python
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import requests

from protein_engine.config import RATE_LIMIT_DEFAULT, RETRY_BACKOFF


class ApiError(RuntimeError):
    def __init__(self, status_code: int, url: str) -> None:
        super().__init__(f"HTTP {status_code} for {url}")
        self.status_code = status_code
        self.url = url


class CacheMissError(RuntimeError):
    pass


class ApiCache:
    def __init__(
        self,
        path: Path,
        transport: object | None = None,
        rate_limit: float = RATE_LIMIT_DEFAULT,
        backoff: Sequence[float] = RETRY_BACKOFF,
    ) -> None:
        self._transport = transport if transport is not None else requests
        self._rate_limit = rate_limit
        self._backoff = list(backoff)
        self._last_request = 0.0
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS http_cache ("
            "key TEXT PRIMARY KEY, body TEXT NOT NULL, fetched_at TEXT NOT NULL)"
        )
        self._db.commit()

    @staticmethod
    def _key(url: str, params: dict | None) -> str:
        canonical = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((url + "\x00" + canonical).encode()).hexdigest()

    def get(self, url: str, params: dict | None = None) -> str | None:
        row = self._db.execute(
            "SELECT body FROM http_cache WHERE key = ?", (self._key(url, params),)
        ).fetchone()
        return row[0] if row else None

    def set(self, url: str, params: dict | None, body: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO http_cache (key, body, fetched_at) VALUES (?, ?, ?)",
            (self._key(url, params), body, datetime.now(timezone.utc).isoformat()),
        )
        self._db.commit()

    def _throttle(self) -> None:
        if self._rate_limit <= 0:
            return
        delay = (1.0 / self._rate_limit) - (time.monotonic() - self._last_request)
        if delay > 0:
            time.sleep(delay)
        self._last_request = time.monotonic()

    def fetch(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        allow_network: bool = True,
    ) -> str:
        cached = self.get(url, params)
        if cached is not None:
            return cached
        if not allow_network:
            raise CacheMissError(f"no cached response for {url} {params}")
        attempts = self._backoff or [0.0]
        last_err: ApiError | None = None
        for delay in attempts:
            self._throttle()
            resp = self._transport.get(url, params=params, headers=headers)
            if resp.status_code == 200:
                self.set(url, params, resp.text)
                return resp.text
            if resp.status_code not in (429, 500, 502, 503, 504):
                raise ApiError(resp.status_code, url)
            last_err = ApiError(resp.status_code, url)
            if delay:
                time.sleep(delay)
        raise last_err if last_err is not None else ApiError(0, url)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/protein_engine/test_api_cache.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/protein_engine/retrieval/api_cache.py tests/protein_engine/conftest.py tests/protein_engine/test_api_cache.py
git commit -m "feat: add shared ApiCache with rate limit, retry, and sqlite cache"
```

---

### Task 4: Taxonomy resolver

**Files:**
- Create: `src/protein_engine/taxonomy/resolver.py`
- Test: `tests/protein_engine/test_resolver.py`

**Interfaces:**
- Consumes: `config` (`NCBI_ESEARCH`, `NCBI_ESUMMARY`), `ApiCache`, `TaxonomyCache`.
- Produces:
  - `ResolutionError` — `ValueError` subclass.
  - `normalize_name(species: str) -> str` (lowercase, collapse whitespace).
  - `resolve(species: str, cache: ApiCache, taxonomy_cache: TaxonomyCache | None = None) -> dict` returning `{"taxonomy_id": int, "resolved_name": str}`. Raises `ResolutionError` on empty name or no NCBI hit.

- [ ] **Step 1: Write the failing test**

Create `tests/protein_engine/test_resolver.py`:

```python
from __future__ import annotations

import pytest

from protein_engine.retrieval.api_cache import ApiCache
from protein_engine.taxonomy.cache import TaxonomyCache
from protein_engine.taxonomy.resolver import ResolutionError, normalize_name, resolve

from conftest import FakeTransport, json_response


def _cache(path, transport, **kwargs):
    return ApiCache(path, transport=transport, rate_limit=0.0, backoff=[], **kwargs)


def test_normalize_name() -> None:
    assert normalize_name("  Bacillus   subtilis ") == "bacillus subtilis"


def test_resolve_via_ncbi_and_cache(tmp_path) -> None:
    transport = FakeTransport(
        [
            json_response({"esearchresult": {"idlist": ["224308"]}}),
            json_response({"result": {"224308": {"scientificname": "Bacillus subtilis subsp. subtilis"}}}),
        ]
    )
    cache = _cache(tmp_path / "http.db", transport)
    tcache = TaxonomyCache(tmp_path / "taxonomy.db")
    assert resolve("Bacillus subtilis", cache, tcache) == {
        "taxonomy_id": 224308,
        "resolved_name": "Bacillus subtilis subsp. subtilis",
    }
    assert tcache.get("bacillus subtilis") == {
        "taxonomy_id": 224308,
        "resolved_name": "Bacillus subtilis subsp. subtilis",
    }


def test_resolve_second_call_hits_taxonomy_cache(tmp_path) -> None:
    tcache = TaxonomyCache(tmp_path / "taxonomy.db")
    tcache.set("bacillus subtilis", {"taxonomy_id": 224308, "resolved_name": "Bacillus subtilis"})
    cache = _cache(tmp_path / "http.db", FakeTransport([]))
    assert resolve("  Bacillus subtilis  ", cache, tcache) == {
        "taxonomy_id": 224308,
        "resolved_name": "Bacillus subtilis",
    }


def test_resolve_raises_when_unknown(tmp_path) -> None:
    transport = FakeTransport([json_response({"esearchresult": {"idlist": []}})])
    cache = _cache(tmp_path / "http.db", transport)
    with pytest.raises(ResolutionError):
        resolve("Unknownus mysteryii", cache)


def test_resolve_raises_on_empty(tmp_path) -> None:
    cache = _cache(tmp_path / "http.db", FakeTransport([]))
    with pytest.raises(ResolutionError):
        resolve("   ", cache)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/protein_engine/test_resolver.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/protein_engine/taxonomy/resolver.py`:

```python
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
    body = cache.fetch(
        NCBI_ESEARCH, params={"db": "taxonomy", "term": f'"{name}"[All Names]', "retmode": "json"}
    )
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/protein_engine/test_resolver.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/protein_engine/taxonomy/resolver.py tests/protein_engine/test_resolver.py
git commit -m "feat: add NCBI taxonomy resolver with cache"
```

---

### Task 5: UniProt client (reviewed-first, paginated)

**Files:**
- Create: `src/protein_engine/retrieval/uniprot_client.py`
- Test: `tests/protein_engine/test_uniprot_client.py`

**Interfaces:**
- Consumes: `config` (`UNIPROT_SEARCH`, `UNIPROT_FIELDS`, `PAGE_SIZE`, `REVIEWED_MIN_COUNT`), `ApiCache`.
- Produces:
  - `fetch_proteins(taxonomy_id: int, cache: ApiCache, reviewed_min_count: int = REVIEWED_MIN_COUNT) -> list[dict]`
  - `normalize(result: dict, reviewed: bool) -> dict`
  - Normalized record keys: `accession`, `protein_name`, `gene`, `length`, `sequence`, `reviewed`, `evidence` (`"reviewed"|"unreviewed"`), `subcellular_location`, `function`, `pdb_structures` (list), `keywords` (list).

- [ ] **Step 1: Write the failing test**

Create `tests/protein_engine/test_uniprot_client.py`:

```python
from __future__ import annotations

from protein_engine.retrieval.api_cache import ApiCache
from protein_engine.retrieval.uniprot_client import fetch_proteins, normalize

from conftest import FakeTransport, json_response


def _cache(path, transport):
    return ApiCache(path, transport=transport, rate_limit=0.0, backoff=[])


def _result(accession: str, gene: str) -> dict:
    return {
        "primaryAccession": accession,
        "proteinDescription": {"recommendedName": {"fullName": {"value": f"{gene} protein"}}},
        "genes": [{"geneName": {"value": gene}}],
        "sequence": {"value": "M" * 100, "length": 100},
        "comments": [{"commentType": "FUNCTION", "texts": [{"value": "does things"}]}],
        "subcellularLocations": [{"location": {"value": "Cytoplasm"}}],
        "uniProtKBCrossReferences": [{"database": "PDB", "id": "1FSE"}],
        "keywords": [{"name": "Sporulation"}],
    }


def test_normalize_maps_fields() -> None:
    rec = normalize(_result("P37476", "spo0A"), reviewed=True)
    assert rec["accession"] == "P37476"
    assert rec["gene"] == "spo0A"
    assert rec["protein_name"] == "spo0A protein"
    assert rec["sequence"] == "M" * 100
    assert rec["length"] == 100
    assert rec["reviewed"] is True
    assert rec["evidence"] == "reviewed"
    assert rec["subcellular_location"] == "Cytoplasm"
    assert rec["function"] == "does things"
    assert rec["pdb_structures"] == ["1FSE"]
    assert rec["keywords"] == ["Sporulation"]


def test_fetch_proteins_reviewed_first(tmp_path) -> None:
    transport = FakeTransport([json_response({"results": [_result("P37476", "spo0A")]})])
    cache = _cache(tmp_path / "http.db", transport)
    proteins = fetch_proteins(224308, cache, reviewed_min_count=1)
    assert len(proteins) == 1
    assert proteins[0]["accession"] == "P37476"
    assert "reviewed:true" in transport.calls[0]["params"]["query"]


def test_fetch_proteins_falls_back_to_unreviewed_when_too_few(tmp_path) -> None:
    transport = FakeTransport([json_response({"results": [_result("P0A", "x")]})])
    cache = _cache(tmp_path / "http.db", transport)
    proteins = fetch_proteins(224308, cache, reviewed_min_count=10)
    assert proteins[0]["evidence"] == "unreviewed"
    assert "reviewed:false" in transport.calls[1]["params"]["query"]


def test_fetch_proteins_walks_pagination(tmp_path) -> None:
    first = {"results": [_result("P1", "a")], "nextLink": "https://rest.uniprot.org/uniprotkb/search?cursor=A"}
    second = {"results": [_result("P2", "b")]}
    transport = FakeTransport([json_response(first), json_response(second)])
    cache = _cache(tmp_path / "http.db", transport)
    proteins = fetch_proteins(224308, cache, reviewed_min_count=2)
    assert [p["accession"] for p in proteins] == ["P1", "P2"]
    assert transport.calls[1]["params"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/protein_engine/test_uniprot_client.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/protein_engine/retrieval/uniprot_client.py`:

```python
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


def fetch_proteins(
    taxonomy_id: int, cache: ApiCache, reviewed_min_count: int = REVIEWED_MIN_COUNT
) -> list[dict]:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/protein_engine/test_uniprot_client.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/protein_engine/retrieval/uniprot_client.py tests/protein_engine/test_uniprot_client.py
git commit -m "feat: add UniProt protein client with reviewed-first retrieval"
```

---

### Task 6: FASTA parser

**Files:**
- Create: `src/protein_engine/sequence/fasta_parser.py`
- Test: `tests/protein_engine/test_fasta_parser.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `parse(text: str) -> list[dict]` — records `{"accession": str, "header": str, "sequence": str}` (accession = first token of header, split on `|`).
  - `to_fasta(proteins: list[dict], resolved_name: str) -> str` — 60-char wrapped FASTA; header `accession|gene|protein_name|resolved_name` with empty fields dropped.
  - `_wrap(sequence: str, width: int = 60) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/protein_engine/test_fasta_parser.py`:

```python
from __future__ import annotations

from protein_engine.sequence.fasta_parser import parse, to_fasta


def test_parse_multi_record() -> None:
    text = ">P37476|spo0A|Stage 0 sporulation protein A\nMNNKILV\n>P0A|gene2\nACDEFG\n"
    records = parse(text)
    assert len(records) == 2
    assert records[0]["accession"] == "P37476"
    assert records[0]["header"] == "P37476|spo0A|Stage 0 sporulation protein A"
    assert records[0]["sequence"] == "MNNKILV"
    assert records[1]["accession"] == "P0A"
    assert records[1]["sequence"] == "ACDEFG"


def test_parse_ncbi_style_accession() -> None:
    records = parse(">sp|P37476|SP0A_BACSU Stage 0 sporulation protein A\nMNNK\n")
    assert records[0]["accession"] == "P37476"


def test_to_fasta_wraps_and_builds_header() -> None:
    proteins = [{"accession": "P37476", "gene": "spo0A", "protein_name": "Spo0A", "sequence": "M" * 130}]
    text = to_fasta(proteins, "Bacillus subtilis")
    assert text.startswith(">P37476|spo0A|Spo0A|Bacillus subtilis\n")
    lines = text.splitlines()
    assert len(lines) == 4  # header + 60 + 60 + 10
    assert len(lines[1]) == 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/protein_engine/test_fasta_parser.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/protein_engine/sequence/fasta_parser.py`:

```python
from __future__ import annotations


def parse(text: str) -> list[dict]:
    records: list[dict] = []
    header: str | None = None
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append(_record(header, "".join(lines)))
            header = line[1:].strip()
            lines = []
        else:
            lines.append(line)
    if header is not None:
        records.append(_record(header, "".join(lines)))
    return records


def _record(header: str, sequence: str) -> dict:
    first = header.split()[0] if header else ""
    accession = first.split("|")[0]
    return {"accession": accession, "header": header, "sequence": sequence}


def to_fasta(proteins: list[dict], resolved_name: str) -> str:
    chunks: list[str] = []
    for protein in proteins:
        fields = [
            protein.get("accession"),
            protein.get("gene"),
            protein.get("protein_name"),
            resolved_name,
        ]
        header = "|".join(str(f) for f in fields if f)
        chunks.append(f">{header}\n{_wrap(protein.get("sequence") or "")}")
    return "\n".join(chunks) + ("\n" if chunks else "")


def _wrap(sequence: str, width: int = 60) -> str:
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/protein_engine/test_fasta_parser.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/protein_engine/sequence/fasta_parser.py tests/protein_engine/test_fasta_parser.py
git commit -m "feat: add FASTA parser and exporter helper"
```

---

### Task 7: NCBI fallback client

**Files:**
- Create: `src/protein_engine/retrieval/ncbi_client.py`
- Test: `tests/protein_engine/test_ncbi_client.py`

**Interfaces:**
- Consumes: `config` (`NCBI_ESEARCH`, `NCBI_EFETCH`), `ApiCache`, `fasta_parser.parse`.
- Produces:
  - `fetch_proteins(taxonomy_id: int, cache: ApiCache, retmax: int = 200) -> list[dict]` — normalized records (same keys as UniProt; `reviewed=False`, `evidence="unreviewed"`, gene/function/etc. `None`).

- [ ] **Step 1: Write the failing test**

Create `tests/protein_engine/test_ncbi_client.py`:

```python
from __future__ import annotations

from protein_engine.retrieval.api_cache import ApiCache
from protein_engine.retrieval.ncbi_client import fetch_proteins

from conftest import FakeResponse, FakeTransport, json_response


def test_ncbi_fetch_parses_fasta(tmp_path) -> None:
    fasta = ">sp|P37476|SP0A_BACSU Stage 0 sporulation protein A\nMNNKILV\n>gi|123|hypothetical\nACDEFG\n"
    transport = FakeTransport(
        [json_response({"esearchresult": {"idlist": ["1", "2"]}}), FakeResponse(fasta)]
    )
    cache = ApiCache(tmp_path / "http.db", transport=transport, rate_limit=0.0, backoff=[])
    proteins = fetch_proteins(224308, cache)
    assert len(proteins) == 2
    assert proteins[0]["accession"] == "P37476"
    assert proteins[0]["sequence"] == "MNNKILV"
    assert proteins[0]["reviewed"] is False
    assert proteins[0]["evidence"] == "unreviewed"
    assert "txid224308[Organism]" in transport.calls[0]["params"]["term"]


def test_ncbi_fetch_empty_when_no_ids(tmp_path) -> None:
    transport = FakeTransport([json_response({"esearchresult": {"idlist": []}})])
    cache = ApiCache(tmp_path / "http.db", transport=transport, rate_limit=0.0, backoff=[])
    assert fetch_proteins(224308, cache) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/protein_engine/test_ncbi_client.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/protein_engine/retrieval/ncbi_client.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/protein_engine/test_ncbi_client.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/protein_engine/retrieval/ncbi_client.py tests/protein_engine/test_ncbi_client.py
git commit -m "feat: add NCBI protein fallback client"
```

---

### Task 8: PDB structure cross-ref client

**Files:**
- Create: `src/protein_engine/retrieval/pdb_client.py`
- Test: `tests/protein_engine/test_pdb_client.py`

**Interfaces:**
- Consumes: `config` (`PDB_UNIPROT`), `ApiCache`, `ApiError`.
- Produces:
  - `has_structure(accession: str, cache: ApiCache) -> bool` — `False` on 404, propagates other `ApiError`s.

- [ ] **Step 1: Write the failing test**

Create `tests/protein_engine/test_pdb_client.py`:

```python
from __future__ import annotations

import pytest

from protein_engine.retrieval.api_cache import ApiCache, ApiError
from protein_engine.retrieval.pdb_client import has_structure

from conftest import FakeResponse, FakeTransport, json_response


def test_has_structure_true(tmp_path) -> None:
    transport = FakeTransport([json_response([{"struct_id": "1FSE"}])])
    cache = ApiCache(tmp_path / "http.db", transport=transport, rate_limit=0.0, backoff=[])
    assert has_structure("P37476", cache) is True


def test_has_structure_false_on_404(tmp_path) -> None:
    transport = FakeTransport([FakeResponse("", status_code=404)])
    cache = ApiCache(tmp_path / "http.db", transport=transport, rate_limit=0.0, backoff=[])
    assert has_structure("P0A1B2", cache) is False


def test_has_structure_propagates_other_errors(tmp_path) -> None:
    transport = FakeTransport([FakeResponse("", status_code=500), FakeResponse("", status_code=500)])
    cache = ApiCache(tmp_path / "http.db", transport=transport, rate_limit=0.0, backoff=[0.0])
    with pytest.raises(ApiError) as excinfo:
        has_structure("P0A1B2", cache)
    assert excinfo.value.status_code == 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/protein_engine/test_pdb_client.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/protein_engine/retrieval/pdb_client.py`:

```python
from __future__ import annotations

import json

from protein_engine.config import PDB_UNIPROT
from protein_engine.retrieval.api_cache import ApiCache, ApiError


def has_structure(accession: str, cache: ApiCache) -> bool:
    try:
        body = cache.fetch(f"{PDB_UNIPROT}/{accession}")
    except ApiError as exc:
        if exc.status_code == 404:
            return False
        raise
    return bool(json.loads(body))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/protein_engine/test_pdb_client.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/protein_engine/retrieval/pdb_client.py tests/protein_engine/test_pdb_client.py
git commit -m "feat: add PDB structure cross-ref client"
```

---

### Task 9: Curated table loaders (essential / virulence / resistance)

**Files:**
- Create: `src/protein_engine/ranking/_common.py`
- Create: `src/protein_engine/ranking/essential.py`
- Create: `src/protein_engine/ranking/virulence.py`
- Create: `src/protein_engine/ranking/resistance.py`
- Test: `tests/protein_engine/test_loaders.py`

**Interfaces:**
- Consumes: nothing external (stdlib `csv`, `dataclasses`).
- Produces:
  - `CuratedTable` dataclass with `.name: str`, `.genes: frozenset[str]`, `.accessions: frozenset[str]`, and `.matches(protein: dict) -> bool` (gene name OR accession match, lowercased).
  - `essential.load_essential_genes(path: Path) -> CuratedTable`
  - `virulence.load_virulence_genes(path: Path) -> CuratedTable`
  - `resistance.load_resistance_genes(path: Path) -> CuratedTable`
  - CSV schema: header row; gene columns `gene, gene_symbol, symbol`; accession columns `accession, uniprot`.

- [ ] **Step 1: Write the failing test**

Create `tests/protein_engine/test_loaders.py`:

```python
from __future__ import annotations

from protein_engine.ranking.essential import load_essential_genes
from protein_engine.ranking.resistance import load_resistance_genes
from protein_engine.ranking.virulence import load_virulence_genes


def _write(path, text: str) -> None:
    path.write_text(text)


def test_essential_matches_gene_or_accession(tmp_path) -> None:
    p = tmp_path / "essential.csv"
    _write(p, "gene,accession\nspoA,\n,P37476\n")
    table = load_essential_genes(p)
    assert table.name == "essential"
    assert table.matches({"gene": "spoA", "accession": "X"}) is True
    assert table.matches({"gene": "zap", "accession": "p37476"}) is True
    assert table.matches({"gene": "zap", "accession": "X"}) is False


def test_virulence_and_resistance_share_columns(tmp_path) -> None:
    vp = tmp_path / "vf.csv"
    rp = tmp_path / "card.csv"
    _write(vp, "gene_symbol\nhlyA\n")
    _write(rp, "gene\ntetA\n")
    assert load_virulence_genes(vp).matches({"gene": "hlya", "accession": "X"}) is True
    assert load_resistance_genes(rp).matches({"gene": "teta", "accession": "X"}) is True


def test_missing_gene_columns_produce_empty_table(tmp_path) -> None:
    p = tmp_path / "empty.csv"
    _write(p, "gene\n")
    table = load_essential_genes(p)
    assert table.matches({"gene": "x", "accession": "y"}) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/protein_engine/test_loaders.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/protein_engine/ranking/_common.py`:

```python
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CuratedTable:
    name: str
    genes: frozenset[str]
    accessions: frozenset[str]

    def matches(self, protein: dict) -> bool:
        gene = (protein.get("gene") or "").strip().lower()
        accession = (protein.get("accession") or "").strip().lower()
        return gene in self.genes or accession in self.accessions


def load_curated_table(
    path: Path,
    name: str,
    gene_columns: Sequence[str],
    accession_columns: Sequence[str],
) -> CuratedTable:
    genes: set[str] = set()
    accessions: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for column in gene_columns:
                value = (row.get(column) or "").strip().lower()
                if value:
                    genes.add(value)
            for column in accession_columns:
                value = (row.get(column) or "").strip().lower()
                if value:
                    accessions.add(value)
    return CuratedTable(name=name, genes=frozenset(genes), accessions=frozenset(accessions))
```

Create `src/protein_engine/ranking/essential.py`:

```python
from __future__ import annotations

from pathlib import Path

from protein_engine.ranking._common import CuratedTable, load_curated_table

GENE_COLUMNS = ("gene", "gene_symbol", "symbol")
ACCESSION_COLUMNS = ("accession", "uniprot")


def load_essential_genes(path: Path) -> CuratedTable:
    return load_curated_table(path, "essential", GENE_COLUMNS, ACCESSION_COLUMNS)
```

Create `src/protein_engine/ranking/virulence.py`:

```python
from __future__ import annotations

from pathlib import Path

from protein_engine.ranking._common import CuratedTable, load_curated_table
from protein_engine.ranking.essential import ACCESSION_COLUMNS, GENE_COLUMNS


def load_virulence_genes(path: Path) -> CuratedTable:
    return load_curated_table(path, "virulence", GENE_COLUMNS, ACCESSION_COLUMNS)
```

Create `src/protein_engine/ranking/resistance.py`:

```python
from __future__ import annotations

from pathlib import Path

from protein_engine.ranking._common import CuratedTable, load_curated_table
from protein_engine.ranking.essential import ACCESSION_COLUMNS, GENE_COLUMNS


def load_resistance_genes(path: Path) -> CuratedTable:
    return load_curated_table(path, "resistance", GENE_COLUMNS, ACCESSION_COLUMNS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/protein_engine/test_loaders.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/protein_engine/ranking/_common.py src/protein_engine/ranking/essential.py src/protein_engine/ranking/virulence.py src/protein_engine/ranking/resistance.py tests/protein_engine/test_loaders.py
git commit -m "feat: add curated gene table loaders for ranking"
```

---

### Task 10: Ranking scorer

**Files:**
- Create: `src/protein_engine/ranking/scorer.py`
- Test: `tests/protein_engine/test_scorer.py`

**Interfaces:**
- Consumes: `config` (`RANKING_WEIGHTS`, `DEFAULT_TOP_N`, `PRESELECT_LIMIT`), `CuratedTable`.
- Produces:
  - `score_terms(protein: dict, curated: dict | None = None, has_structure: bool = False, weights: dict | None = None) -> dict` — keys `essential, virulence, resistance, evidence, structure` in [0,1].
  - `compute_score(terms: dict, weights: dict | None = None) -> float`
  - `preselect(proteins: list[dict], curated: dict | None = None, limit: int = PRESELECT_LIMIT) -> list[dict]` — scored, sorted desc, structure term 0.
  - `rank(proteins: list[dict], structure_flags: dict | None = None, curated: dict | None = None, weights: dict | None = None, top_n: int = DEFAULT_TOP_N) -> list[dict]` — sorted by `(-score, accession)`, each record gains `score` (rounded 3) and `score_breakdown`.
  - `curated` is `{"essential": CuratedTable, "virulence": CuratedTable, "resistance": CuratedTable}`.

- [ ] **Step 1: Write the failing test**

Create `tests/protein_engine/test_scorer.py`:

```python
from __future__ import annotations

from protein_engine.ranking._common import CuratedTable
from protein_engine.ranking.scorer import compute_score, preselect, rank, score_terms


def _tables() -> dict:
    return {
        "essential": CuratedTable("essential", frozenset({"spoa"}), frozenset()),
        "virulence": CuratedTable("virulence", frozenset(), frozenset()),
        "resistance": CuratedTable("resistance", frozenset(), frozenset()),
    }


def _protein(accession: str, gene: str, reviewed: bool) -> dict:
    return {"accession": accession, "gene": gene, "reviewed": reviewed, "sequence": "M" * 100}


def test_score_terms_essential_evidence_structure() -> None:
    terms = score_terms(_protein("P37476", "spoA", True), curated=_tables(), has_structure=True)
    assert terms == {"essential": 1.0, "virulence": 0.0, "resistance": 0.0, "evidence": 1.0, "structure": 1.0}


def test_score_terms_unreviewed_evidence_is_half() -> None:
    terms = score_terms(_protein("P0A", "x", False), curated=_tables())
    assert terms["evidence"] == 0.5


def test_compute_score_uses_weights() -> None:
    terms = {"essential": 1.0, "virulence": 0.0, "resistance": 0.0, "evidence": 1.0, "structure": 1.0}
    assert compute_score(terms) == 0.55


def test_rank_orders_and_injects_breakdown() -> None:
    proteins = [_protein("P37476", "spoA", True), _protein("P0A", "x", False)]
    ranked = rank(proteins, structure_flags={"P37476": True}, curated=_tables(), top_n=2)
    assert ranked[0]["accession"] == "P37476"
    assert ranked[0]["score"] == 0.55
    assert ranked[0]["score_breakdown"]["essential"] == 1.0
    assert ranked[1]["score"] == round(0.075, 3)


def test_rank_honors_top_n() -> None:
    proteins = [_protein("P1", "a", True), _protein("P2", "b", True)]
    ranked = rank(proteins, top_n=1)
    assert len(ranked) == 1


def test_preselect_excludes_structure_term() -> None:
    proteins = [_protein("P37476", "spoA", True)]
    selected = preselect(proteins, curated=_tables(), limit=10)
    assert selected[0]["score"] == 0.45  # 0.30 essential + 0.15 evidence, no structure
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/protein_engine/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/protein_engine/ranking/scorer.py`:

```python
from __future__ import annotations

from protein_engine.config import DEFAULT_TOP_N, PRESELECT_LIMIT, RANKING_WEIGHTS
from protein_engine.ranking._common import CuratedTable


def score_terms(
    protein: dict,
    curated: dict | None = None,
    has_structure: bool = False,
    weights: dict | None = None,
) -> dict:
    weights = weights or RANKING_WEIGHTS
    curated = curated or {}
    essential_table: CuratedTable | None = curated.get("essential")
    virulence_table: CuratedTable | None = curated.get("virulence")
    resistance_table: CuratedTable | None = curated.get("resistance")
    return {
        "essential": 1.0 if essential_table and essential_table.matches(protein) else 0.0,
        "virulence": 1.0 if virulence_table and virulence_table.matches(protein) else 0.0,
        "resistance": 1.0 if resistance_table and resistance_table.matches(protein) else 0.0,
        "evidence": 1.0 if protein.get("reviewed") else 0.5,
        "structure": 1.0 if has_structure else 0.0,
    }


def compute_score(terms: dict, weights: dict | None = None) -> float:
    weights = weights or RANKING_WEIGHTS
    return sum(weights[key] * terms[key] for key in weights)


def _scored(
    proteins: list[dict],
    curated: dict | None,
    structure_flags: dict,
    weights: dict | None = None,
) -> list[dict]:
    flags = structure_flags or {}
    scored = []
    for protein in proteins:
        terms = score_terms(
            protein,
            curated=curated,
            has_structure=flags.get(protein.get("accession"), False),
            weights=weights,
        )
        scored.append({**protein, "score": round(compute_score(terms, weights), 3), "score_breakdown": terms})
    scored.sort(key=lambda item: (-item["score"], item["accession"]))
    return scored


def preselect(proteins: list[dict], curated: dict | None = None, limit: int = PRESELECT_LIMIT) -> list[dict]:
    return _scored(proteins, curated, {})[:limit]


def rank(
    proteins: list[dict],
    structure_flags: dict | None = None,
    curated: dict | None = None,
    weights: dict | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict]:
    return _scored(proteins, curated, structure_flags, weights)[:top_n]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/protein_engine/test_scorer.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/protein_engine/ranking/scorer.py tests/protein_engine/test_scorer.py
git commit -m "feat: add weighted ranking scorer"
```

---

### Task 11: Sequence validator

**Files:**
- Create: `src/protein_engine/sequence/validator.py`
- Test: `tests/protein_engine/test_validator.py`

**Interfaces:**
- Consumes: `config` (`MIN_SEQUENCE_LENGTH`).
- Produces:
  - `AA_ALPHABET` — frozenset of the 20 standard amino acids.
  - `validate(protein: dict, min_length: int = MIN_SEQUENCE_LENGTH) -> dict` — `{"ok": bool, "protein": dict, "issues": list[str]}`; issues like `"too_short"`, `"non_standard:XZ"`.
  - `dedupe_by_accession(proteins: list[dict]) -> list[dict]` — first occurrence wins.
  - `split_valid(proteins: list[dict], min_length: int = MIN_SEQUENCE_LENGTH) -> tuple[list[dict], list[dict]]` — valid list + excluded list (each excluded record gains `excluded_reasons`).

- [ ] **Step 1: Write the failing test**

Create `tests/protein_engine/test_validator.py`:

```python
from __future__ import annotations

from protein_engine.sequence.validator import dedupe_by_accession, split_valid, validate


def test_validate_ok() -> None:
    result = validate({"accession": "P1", "sequence": "M" * 30})
    assert result["ok"] is True
    assert result["issues"] == []


def test_validate_too_short() -> None:
    result = validate({"accession": "P1", "sequence": "M" * 10})
    assert result["ok"] is False
    assert "too_short" in result["issues"]


def test_validate_non_standard_residues() -> None:
    result = validate({"accession": "P1", "sequence": "M" * 30 + "XZ"})
    assert result["ok"] is False
    assert any(issue.startswith("non_standard:") for issue in result["issues"])


def test_dedupe_by_accession_keeps_first() -> None:
    proteins = [{"accession": "P1"}, {"accession": "P2"}, {"accession": "P1"}]
    assert [p["accession"] for p in dedupe_by_accession(proteins)] == ["P1", "P2"]


def test_split_valid_partitions() -> None:
    good = {"accession": "P1", "sequence": "M" * 30}
    bad = {"accession": "P2", "sequence": "Z" * 30}
    valid, excluded = split_valid([good, bad, good])
    assert [p["accession"] for p in valid] == ["P1"]
    assert [p["accession"] for p in excluded] == ["P2"]
    assert "excluded_reasons" in excluded[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/protein_engine/test_validator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/protein_engine/sequence/validator.py`:

```python
from __future__ import annotations

from protein_engine.config import MIN_SEQUENCE_LENGTH

AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


def validate(protein: dict, min_length: int = MIN_SEQUENCE_LENGTH) -> dict:
    issues: list[str] = []
    sequence = protein.get("sequence") or ""
    if len(sequence) < min_length:
        issues.append("too_short")
    non_standard = sorted({residue for residue in sequence.upper() if residue not in AA_ALPHABET})
    if non_standard:
        issues.append(f"non_standard:{''.join(non_standard)}")
    return {"ok": not issues, "protein": protein, "issues": issues}


def dedupe_by_accession(proteins: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for protein in proteins:
        accession = protein.get("accession")
        if accession in seen:
            continue
        seen.add(accession)
        result.append(protein)
    return result


def split_valid(
    proteins: list[dict], min_length: int = MIN_SEQUENCE_LENGTH
) -> tuple[list[dict], list[dict]]:
    valid: list[dict] = []
    excluded: list[dict] = []
    for protein in dedupe_by_accession(proteins):
        check = validate(protein, min_length)
        if check["ok"]:
            valid.append(protein)
        else:
            excluded.append({**protein, "excluded_reasons": check["issues"]})
    return valid, excluded
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/protein_engine/test_validator.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/protein_engine/sequence/validator.py tests/protein_engine/test_validator.py
git commit -m "feat: add sequence validator with dedupe and partition"
```

---

### Task 12: Exporter (JSON + FASTA output contract)

**Files:**
- Create: `src/protein_engine/sequence/exporter.py`
- Test: `tests/protein_engine/test_exporter.py`

**Interfaces:**
- Consumes: `fasta_parser.to_fasta`, `datetime`.
- Produces:
  - `_slugify(name: str) -> str`
  - `write(species: str, resolved_name: str, taxonomy_id: int, proteins: list[dict], excluded: list[dict], output_dir: Path | str, source: str = "uniprot") -> Path` — writes `proteins.json` (contract from spec §7) + `sequences.fasta` into `output_dir/<slug>/`, returns that dir.

- [ ] **Step 1: Write the failing test**

Create `tests/protein_engine/test_exporter.py`:

```python
from __future__ import annotations

import json

from protein_engine.sequence.exporter import write


def _protein() -> dict:
    return {
        "rank": 1,
        "accession": "P37476",
        "gene": "spo0A",
        "protein_name": "Spo0A",
        "length": 60,
        "sequence": "M" * 60,
        "score": 0.55,
        "score_breakdown": {"essential": 1.0, "virulence": 0.0, "resistance": 0.0, "evidence": 1.0, "structure": 0.0},
        "evidence": "reviewed",
        "has_structure": False,
        "subcellular_location": "Cytoplasm",
        "function": "response regulator",
    }


def test_write_produces_contract(tmp_path) -> None:
    proteins = [_protein()]
    out = write(
        "Bacillus subtilis",
        "Bacillus subtilis subsp. subtilis",
        224308,
        proteins,
        [],
        tmp_path,
        source="uniprot",
    )
    assert out == tmp_path / "bacillus_subtilis_subsp_subtilis"
    payload = json.loads((out / "proteins.json").read_text())
    assert payload["species"] == "Bacillus subtilis"
    assert payload["taxonomy_id"] == 224308
    assert payload["resolved_name"] == "Bacillus subtilis subsp. subtilis"
    assert payload["source"] == "uniprot"
    assert payload["query_timestamp"]
    assert payload["excluded"] == []
    assert payload["selected_proteins"][0]["accession"] == "P37476"
    assert "sequence" in payload["selected_proteins"][0]
    fasta = (out / "sequences.fasta").read_text()
    assert fasta.startswith(">P37476|spo0A|Spo0A|Bacillus subtilis subsp. subtilis\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/protein_engine/test_exporter.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/protein_engine/sequence/exporter.py`:

```python
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from protein_engine.sequence.fasta_parser import to_fasta


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"


def write(
    species: str,
    resolved_name: str,
    taxonomy_id: int,
    proteins: list[dict],
    excluded: list[dict],
    output_dir: Path | str,
    source: str = "uniprot",
) -> Path:
    out = Path(output_dir) / _slugify(resolved_name or species)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "species": species,
        "taxonomy_id": taxonomy_id,
        "resolved_name": resolved_name,
        "source": source,
        "query_timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_proteins": proteins,
        "excluded": excluded,
    }
    (out / "proteins.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "sequences.fasta").write_text(to_fasta(proteins, resolved_name))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/protein_engine/test_exporter.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/protein_engine/sequence/exporter.py tests/protein_engine/test_exporter.py
git commit -m "feat: add JSON/FASTA exporter matching output contract"
```

---

### Task 13: Pipeline orchestrator

**Files:**
- Create: `src/protein_engine/pipeline.py`
- Test: `tests/protein_engine/test_pipeline.py`

**Interfaces:**
- Consumes: everything above (`config`, `resolver`, `ApiCache`, `TaxonomyCache`, `uniprot_client`, `ncbi_client`, `pdb_client`, `scorer`, `validator`, `exporter`, plus ranking loaders).
- Produces:
  - `load_curated(reference_dir: Path | str) -> dict` — `{"essential": CuratedTable, ...}` for whichever of `essential_genes.csv`, `virulence_genes.csv`, `resistance_genes.csv` exist in the reference dir.
  - `run(species: str, *, top_n: int = DEFAULT_TOP_N, output_dir: Path | str = PATHS.output_dir, cache_dir: Path | str = PATHS.cache_dir, reference_dir: Path | str = PATHS.reference_dir, offline: bool = False, transport: object | None = None, rate_limit: float | None = None) -> dict` — returns the full JSON payload plus `"output_dir"`.

- [ ] **Step 1: Write the failing test**

Create `tests/protein_engine/test_pipeline.py`:

```python
from __future__ import annotations

import json

from protein_engine.pipeline import load_curated, run

from conftest import FakeTransport, json_response


def _uniprot_result(accession: str, gene: str, length: int = 100) -> dict:
    return {
        "primaryAccession": accession,
        "proteinDescription": {"recommendedName": {"fullName": {"value": f"{gene} protein"}}},
        "genes": [{"geneName": {"value": gene}}],
        "sequence": {"value": "M" * length, "length": length},
        "comments": [],
        "subcellularLocations": [],
        "uniProtKBCrossReferences": [],
        "keywords": [],
    }


def test_load_curated_picks_existing_files(tmp_path) -> None:
    (tmp_path / "essential_genes.csv").write_text("gene\nspoA\n")
    tables = load_curated(tmp_path)
    assert "essential" in tables
    assert "virulence" not in tables
    assert tables["essential"].matches({"gene": "spoA", "accession": "x"}) is True


def test_run_end_to_end(tmp_path) -> None:
    reference_dir = tmp_path / "ref"
    reference_dir.mkdir()
    (reference_dir / "essential_genes.csv").write_text("gene\nspoA\n")
    results = [_uniprot_result("P37476", "spoA", 405)]
    results += [_uniprot_result(f"P0A0{str(i).zfill(2)}", f"gene{i}", 300) for i in range(11)]
    pdb_responses = [json_response([{"struct_id": "1FSE"}])] + [json_response([]) for _ in range(11)]
    transport = FakeTransport(
        [
            json_response({"esearchresult": {"idlist": ["224308"]}}),
            json_response({"result": {"224308": {"scientificname": "Bacillus subtilis"}}}),
            json_response({"results": results}),
            *pdb_responses,
        ]
    )
    payload = run(
        "Bacillus subtilis",
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        reference_dir=str(reference_dir),
        transport=transport,
        rate_limit=0.0,
    )
    assert payload["species"] == "Bacillus subtilis"
    assert payload["taxonomy_id"] == 224308
    assert payload["source"] == "uniprot"
    assert payload["selected_proteins"][0]["accession"] == "P37476"
    assert payload["selected_proteins"][0]["score"] == 0.55
    assert payload["output_dir"]
    out_dir = tmp_path / "out" / "bacillus_subtilis"
    assert (out_dir / "proteins.json").exists()
    assert (out_dir / "sequences.fasta").exists()
    stored = json.loads((out_dir / "proteins.json").read_text())
    assert stored["species"] == "Bacillus subtilis"
    assert stored["selected_proteins"][0]["sequence"].startswith("M")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/protein_engine/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/protein_engine/pipeline.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from protein_engine.config import DEFAULT_TOP_N, PATHS, PRESELECT_LIMIT, RATE_LIMIT_DEFAULT, RETRY_BACKOFF
from protein_engine.ranking import essential, resistance, virulence
from protein_engine.ranking.scorer import preselect, rank
from protein_engine.retrieval import ncbi_client, pdb_client, uniprot_client
from protein_engine.retrieval.api_cache import ApiCache
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
    structure_flags = {
        p["accession"]: pdb_client.has_structure(p["accession"], cache)
        for p in shortlist
        if p.get("accession")
    }
    ranked = rank(shortlist, structure_flags=structure_flags, curated=curated, top_n=top_n)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/protein_engine/test_pipeline.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/protein_engine/pipeline.py tests/protein_engine/test_pipeline.py
git commit -m "feat: add protein engine pipeline orchestrator"
```

---

### Task 14: CLI entry point

**Files:**
- Create: `scripts/run_protein_pipeline.py`
- Test: `tests/protein_engine/test_cli.py`

**Interfaces:**
- Consumes: `pipeline.run`, `PATHS`, `ResolutionError`, `ApiError`/`CacheMissError`, and (for `--image`) `bacteria_assistant.config.MODEL_PATH` + `bacteria_assistant.inference.predict_bacteria_image`.
- Produces:
  - `main(argv: list[str] | None = None) -> int` — exit `0` success, `1` infra/network error, `2` user-input/resolution error. Prints JSON payload on success.
  - Args: `--species`, `--image`, `--top-n` (default `DEFAULT_TOP_N`), `--output-dir`, `--cache-dir`, `--reference-dir`, `--offline`.

- [ ] **Step 1: Write the failing test**

Create `tests/protein_engine/test_cli.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "run_protein_pipeline.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )


def test_cli_help_exits_zero() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "--species" in result.stdout


def test_cli_missing_species_and_image_exits_two() -> None:
    result = _run()
    assert result.returncode == 2
    assert "either --species or --image is required" in result.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/protein_engine/test_cli.py -v`
Expected: FAIL with `FileNotFoundError: [Errno 2]` (script missing)

- [ ] **Step 3: Write minimal implementation**

Create `scripts/run_protein_pipeline.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from protein_engine import pipeline
from protein_engine.config import DEFAULT_TOP_N, PATHS
from protein_engine.retrieval.api_cache import ApiError, CacheMissError
from protein_engine.taxonomy.resolver import ResolutionError


def _predict_species(image_path: Path) -> str:
    try:
        from bacteria_assistant.config import MODEL_PATH
        from bacteria_assistant.inference import predict_bacteria_image
    except ImportError as exc:
        raise ResolutionError("CV pipeline unavailable; use --species instead") from exc
    result = predict_bacteria_image(
        image_path=image_path,
        model_path=PROJECT_ROOT / MODEL_PATH,
        mode="basic",
    )
    name = result.get("predicted_bacteria_name")
    if not name or name == "unknown":
        raise ResolutionError(f"could not predict a species from image: {image_path}")
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank folding-ready proteins for a species.")
    parser.add_argument("--species", type=str, help="Scientific species name")
    parser.add_argument("--image", type=Path, help="Petri dish image (predicts species via CV)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Number of proteins to return")
    parser.add_argument("--output-dir", type=Path, default=PATHS.output_dir)
    parser.add_argument("--cache-dir", type=Path, default=PATHS.cache_dir)
    parser.add_argument("--reference-dir", type=Path, default=PATHS.reference_dir)
    parser.add_argument("--offline", action="store_true", help="Serve from cache only")
    args = parser.parse_args(argv)

    if not args.species and args.image is None:
        parser.error("either --species or --image is required")

    try:
        species = args.species or _predict_species(args.image)
        payload = pipeline.run(
            species,
            top_n=args.top_n,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            reference_dir=args.reference_dir,
            offline=args.offline,
        )
    except ResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ApiError, CacheMissError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/protein_engine/test_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_protein_pipeline.py tests/protein_engine/test_cli.py
git commit -m "feat: add protein pipeline CLI entry point"
```

---

### Task 15: Full verification + docs sync

**Files:**
- Modify: `src/protein_engine/documentation/design.md`, `src/protein_engine/documentation/architecture.md` (only if behavior drifted during implementation)

**Interfaces:**
- Consumes: everything.

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: ALL PASS (existing 2 tests + all new protein_engine tests). If any failure: run `pytest tests/protein_engine/ -v` to isolate, fix, rerun.

- [ ] **Step 2: Run linters**

Run: `ruff check .`
Expected: no errors.

Run: `ruff format --check .`
Expected: no files would be reformatted. If a file would be reformatted, run `ruff format <file>` and re-run checks.

- [ ] **Step 3: Update docs if behavior drifted**

Compare implemented signatures against `src/protein_engine/documentation/architecture.md` §6 (interfaces). Update any signatures that drifted (e.g. `pipeline.run` keyword args, `ApiCache` constructor). Commit only if changed.

- [ ] **Step 4: Manual smoke test (optional, needs network)**

Run: `python scripts/run_protein_pipeline.py --species "Bacillus subtilis" --top-n 5 --output-dir output --cache-dir cache`
Expected: prints JSON payload with `selected_proteins`, writes `output/bacillus_subtilis/{proteins.json,sequences.fasta}`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: verify protein engine suite and sync docs"
```

---

## Self-Review Notes

- **Spec coverage:** taxonomy (T2, T4) · UniProt reviewed-first retrieval (T5) · NCBI fallback (T7) · PDB structure flag (T8) · curated ranking (T9, T10) · validation/dedupe (T11) · JSON+FASTA output contract (T12) · orchestration + CLI + exit codes (T13, T14) · caching/rate-limit/retry/offline (T3) · E2E with stubbed UniProt (T13) · deps `requests` only (T1). M5 folding integration is out of v1 scope by design.
- **Type consistency:** `ApiCache.fetch(url, params, headers, allow_network)` used consistently by T4–T8/T13; `CuratedTable.matches(protein)` in T9/T10; scorer `preselect`/`rank` signatures match T13; normalized protein keys shared across T5/T7/T11/T12/T13.
- **Placeholder scan:** every task carries concrete test + implementation code; no TBDs.
- **Deliberate deviations from spec structure:** added `pipeline.py` (orchestration, mirroring `bacteria_assistant/inference.py`, for testability) and `ranking/_common.py` (shared CSV loader) — both internal; documented in plan, noted for docs sync in T15.
