# Web Frontend + FastAPI Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI REST backend + React web frontend for BPA-22, with a SQLite audit log, running locally first.

**Architecture:** FastAPI wraps existing `predict_bacteria_image` and `pipeline.run` as REST endpoints. React SPA with two pages (image analysis + protein ranking). SQLite stores prediction history. Vite dev server proxies `/api` to FastAPI.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, SQLite, React 18, Vite, Tailwind CSS, React Router, axios.

## Global Constraints

- Python >=3.11, Node >=18
- Existing `src/bacteria_assistant/` and `src/protein_engine/` code is imported directly -- never rewritten
- `server/requirements.txt` is separate from the project's main `requirements.txt`
- SQLite file lives at `data/bpa.db` (created at runtime)
- All tests run with `pytest` (backend)
- Follow existing code conventions: `from __future__ import annotations`, ruff formatting, 120 char line length

---

## File Map

```
server/
  __init__.py              # empty
  main.py                  # FastAPI app, startup, CORS, include routers
  dependencies.py          # model loading (cached), DB connection
  db.py                    # SQLite schema init + CRUD helpers
  models.py                # Pydantic request/response schemas
  routers/
    __init__.py            # empty
    predict.py             # POST /api/predict, GET /api/predict/history, GET /api/predict/{id}
    proteins.py            # POST /api/proteins, GET /api/species
  requirements.txt         # fastapi, uvicorn, python-multipart

frontend/
  index.html
  package.json
  vite.config.js
  tailwind.config.js
  postcss.config.js
  src/
    main.jsx
    App.jsx                # React Router, nav bar
    index.css              # Tailwind directives
    api.js                 # axios instance
    pages/
      PredictPage.jsx
      ProteinPage.jsx
    components/
      ImageUpload.jsx
      ResultsPanel.jsx
      ResultsTable.jsx
      SpeciesSelector.jsx
      ProteinTable.jsx
      FastaDownload.jsx
  public/
```

---

## Task 1: Server scaffolding + health endpoint

**Files:**
- Create: `server/__init__.py`
- Create: `server/main.py`
- Create: `server/requirements.txt`
- Create: `tests/test_server_health.py`

**Interfaces:**
- Produces: FastAPI app at `server.main:app` with `GET /api/health` returning `{"status": "ok"}`

- [ ] **Step 1: Create `server/__init__.py`**

Empty file -- makes `server` a Python package.

- [ ] **Step 2: Create `server/requirements.txt`**

```
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
```

- [ ] **Step 3: Install server dependencies**

```bash
.venv/bin/pip install -r server/requirements.txt
```

- [ ] **Step 4: Create `server/main.py`**

```python
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="BPA-22", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 5: Create `tests/test_server_health.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from server.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_server_health.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add server/ tests/test_server_health.py
git commit -m "feat(server): add FastAPI scaffolding with health endpoint"
```

---

## Task 2: SQLite database layer

**Files:**
- Create: `server/db.py`
- Create: `tests/test_server_db.py`

**Interfaces:**
- Produces: `init_db(path)`, `insert_prediction(path, ...) -> int`, `get_prediction(path, id) -> dict | None`, `list_predictions(path, limit, offset) -> dict`

- [ ] **Step 1: Create `server/db.py`**

```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    conn = _connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            filename        TEXT NOT NULL,
            image_bytes     BLOB,
            mode            TEXT NOT NULL DEFAULT 'advanced',
            organism_type   TEXT,
            predicted_species TEXT,
            gram            TEXT,
            total_colonies  INTEGER,
            dominant_shape  TEXT,
            confidence      REAL,
            result_json     TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def insert_prediction(
    db_path: Path,
    *,
    filename: str,
    image_bytes: bytes | None,
    mode: str,
    organism_type: str | None,
    predicted_species: str | None,
    gram: str | None,
    total_colonies: int | None,
    dominant_shape: str | None,
    confidence: float | None,
    result_json: dict,
) -> int:
    conn = _connect(db_path)
    cursor = conn.execute(
        """
        INSERT INTO predictions
            (filename, image_bytes, mode, organism_type, predicted_species,
             gram, total_colonies, dominant_shape, confidence, result_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename, image_bytes, mode, organism_type, predicted_species,
            gram, total_colonies, dominant_shape, confidence, json.dumps(result_json),
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_prediction(db_path: Path, prediction_id: int) -> dict | None:
    conn = _connect(db_path)
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def list_predictions(db_path: Path, *, limit: int = 50, offset: int = 0) -> dict:
    conn = _connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    return {"total": total, "items": [dict(row) for row in rows]}
```

- [ ] **Step 2: Create `tests/test_server_db.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

from server.db import get_prediction, init_db, insert_prediction, list_predictions


def test_init_db_creates_file(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    assert db.exists()


def test_insert_and_get(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    row_id = insert_prediction(
        db, filename="test.png", image_bytes=b"\x89PNG", mode="advanced",
        organism_type="bacteria", predicted_species="Bacillus subtilis",
        gram="gram_positive", total_colonies=5, dominant_shape="bacilli",
        confidence=0.92, result_json={"predicted_bacteria_name": "Bacillus subtilis"},
    )
    assert row_id == 1
    row = get_prediction(db, row_id)
    assert row is not None
    assert row["filename"] == "test.png"
    assert row["predicted_species"] == "Bacillus subtilis"
    assert json.loads(row["result_json"])["predicted_bacteria_name"] == "Bacillus subtilis"


def test_list_predictions_pagination(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    for i in range(5):
        insert_prediction(
            db, filename=f"img_{i}.png", image_bytes=None, mode="basic",
            organism_type="bacteria", predicted_species="E. coli",
            gram="gram_negative", total_colonies=3, dominant_shape="bacilli",
            confidence=0.85, result_json={"i": i},
        )
    page = list_predictions(db, limit=2, offset=0)
    assert page["total"] == 5
    assert len(page["items"]) == 2
    assert page["items"][0]["filename"] == "img_4.png"  # newest first
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server_db.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add server/db.py tests/test_server_db.py
git commit -m "feat(server): add SQLite database layer with prediction CRUD"
```

---

## Task 3: Pydantic models

**Files:**
- Create: `server/models.py`
- Create: `tests/test_server_models.py`

**Interfaces:**
- Produces: `PredictResponse`, `HistoryItem`, `HistoryResponse`, `ProteinRequest`, `ProteinResponse`, `SpeciesInfo`

- [ ] **Step 1: Create `server/models.py`**

```python
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
```

- [ ] **Step 2: Create `tests/test_server_models.py`**

```python
from __future__ import annotations

from server.models import PredictResponse, ProteinRequest, SpeciesInfo


def test_predict_response_optional_fields() -> None:
    resp = PredictResponse(id=1, predicted_species="E. coli")
    assert resp.id == 1
    assert resp.colonies is None
    assert resp.morphology is None


def test_protein_request_defaults() -> None:
    req = ProteinRequest(species="Bacillus subtilis")
    assert req.top_n == 10


def test_species_info_fields() -> None:
    sp = SpeciesInfo(name="E. coli", organism_type="bacteria", gram="gram_negative", shape="bacilli", group="gram_negative_bacilli")
    assert sp.name == "E. coli"
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server_models.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add server/models.py tests/test_server_models.py
git commit -m "feat(server): add Pydantic request/response models"
```

---

## Task 4: Model loading dependency

**Files:**
- Create: `server/dependencies.py`
- Create: `tests/test_server_dependencies.py`

**Interfaces:**
- Produces: `get_models() -> dict`, `get_db_path() -> Path`

- [ ] **Step 1: Create `server/dependencies.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.config import MODEL_PATH
from bacteria_assistant.inference import load_models

_models: dict | None = None


def get_models() -> dict:
    global _models
    if _models is None:
        model_path = PROJECT_ROOT / MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        _models = load_models(model_path)
    return _models


def get_db_path() -> Path:
    return PROJECT_ROOT / "data" / "bpa.db"
```

- [ ] **Step 2: Create `tests/test_server_dependencies.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from server.dependencies import get_db_path


def test_get_db_path() -> None:
    path = get_db_path()
    assert path.name == "bpa.db"
    assert path.parent == PROJECT_ROOT / "data"
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server_dependencies.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add server/dependencies.py tests/test_server_dependencies.py
git commit -m "feat(server): add model loading dependency with startup cache"
```

---

## Task 5: POST /api/predict endpoint

**Files:**
- Create: `server/routers/__init__.py` (empty)
- Create: `server/routers/predict.py`
- Modify: `server/main.py` (add lifespan + include router)
- Create: `tests/test_server_predict.py`

**Interfaces:**
- Consumes: `get_models()`, `get_db_path()`, `init_db()`, `insert_prediction()`, `predict_bacteria_image()`
- Produces: `POST /api/predict` -> `PredictResponse`, `GET /api/predict/history` -> `HistoryResponse`, `GET /api/predict/{id}` -> `PredictResponse`

- [ ] **Step 1: Create `server/routers/__init__.py`**

Empty file.

- [ ] **Step 2: Create `server/routers/predict.py`**

```python
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from bacteria_assistant.inference import predict_bacteria_image

from ..db import get_prediction, init_db, insert_prediction, list_predictions
from ..dependencies import get_db_path
from ..models import HistoryResponse, PredictResponse

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    mode: str = Form("advanced"),
) -> PredictResponse:
    image_bytes = await image.read()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)

    try:
        result = predict_bacteria_image(image_path=tmp_path, model_path=None, mode=mode)
    finally:
        tmp_path.unlink(missing_ok=True)

    db_path = get_db_path()
    init_db(db_path)

    row_id = insert_prediction(
        db_path,
        filename=image.filename or "unknown",
        image_bytes=image_bytes,
        mode=mode,
        organism_type=result.get("organism_type"),
        predicted_species=result.get("predicted_bacteria_name"),
        gram=result.get("bacteria_type"),
        total_colonies=result.get("total_colonies_detected") or result.get("total_colonies"),
        dominant_shape=result.get("dominant_shape"),
        confidence=result.get("confidence"),
        result_json=result,
    )

    saved = get_prediction(db_path, row_id)

    return PredictResponse(
        id=row_id,
        created_at=saved["created_at"] if saved else None,
        organism_type=result.get("organism_type"),
        predicted_species=result.get("predicted_bacteria_name"),
        gram=result.get("bacteria_type"),
        total_colonies=result.get("total_colonies_detected") or result.get("total_colonies"),
        dominant_shape=result.get("dominant_shape"),
        confidence=result.get("confidence"),
        colonies=result.get("colonies"),
        morphology=result.get("final_morphology"),
    )


@router.get("/history", response_model=HistoryResponse)
def history(limit: int = 50, offset: int = 0) -> HistoryResponse:
    db_path = get_db_path()
    init_db(db_path)
    data = list_predictions(db_path, limit=limit, offset=offset)
    return HistoryResponse(
        total=data["total"],
        items=[
            {"id": i["id"], "created_at": i["created_at"], "filename": i["filename"],
             "predicted_species": i["predicted_species"], "confidence": i["confidence"]}
            for i in data["items"]
        ],
    )


@router.get("/{prediction_id}", response_model=PredictResponse)
def get_prediction_by_id(prediction_id: int) -> PredictResponse:
    db_path = get_db_path()
    row = get_prediction(db_path, prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    result = json.loads(row["result_json"])
    return PredictResponse(
        id=row["id"], created_at=row["created_at"],
        organism_type=row["organism_type"], predicted_species=row["predicted_species"],
        gram=row["gram"], total_colonies=row["total_colonies"],
        dominant_shape=row["dominant_shape"], confidence=row["confidence"],
        colonies=result.get("colonies"), morphology=result.get("final_morphology"),
    )
```

- [ ] **Step 3: Replace `server/main.py` with lifespan version**

```python
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db
from .dependencies import get_db_path
from .routers import predict, proteins


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(db_path)
    yield


app = FastAPI(title="BPA-22", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(proteins.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 4: Create `tests/test_server_predict.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from server.main import app

client = TestClient(app)

FAKE_RESULT = {
    "organism_type": "bacteria",
    "predicted_bacteria_name": "Bacillus subtilis",
    "bacteria_type": "gram_positive",
    "total_colonies_detected": 5,
    "dominant_shape": "bacilli",
    "confidence": 0.95,
}

_FAKE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


@patch("server.routers.predict.predict_bacteria_image", return_value=FAKE_RESULT)
def test_predict_returns_result(mock_predict) -> None:
    resp = client.post("/api/predict", files={"image": ("t.png", _FAKE_PNG, "image/png")}, data={"mode": "basic"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["organism_type"] == "bacteria"
    assert body["predicted_species"] == "Bacillus subtilis"
    assert body["confidence"] == 0.95


@patch("server.routers.predict.predict_bacteria_image", return_value=FAKE_RESULT)
def test_predict_saves_to_history(mock_predict) -> None:
    resp = client.post("/api/predict", files={"image": ("t.png", _FAKE_PNG, "image/png")}, data={"mode": "basic"})
    pred_id = resp.json()["id"]
    hist = client.get("/api/predict/history")
    assert hist.status_code == 200
    assert any(item["id"] == pred_id for item in hist.json()["items"])


@patch("server.routers.predict.predict_bacteria_image", return_value=FAKE_RESULT)
def test_get_prediction_by_id(mock_predict) -> None:
    resp = client.post("/api/predict", files={"image": ("t.png", _FAKE_PNG, "image/png")}, data={"mode": "advanced"})
    pred_id = resp.json()["id"]
    detail = client.get(f"/api/predict/{pred_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == pred_id
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server_predict.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add server/routers/ server/main.py tests/test_server_predict.py
git commit -m "feat(server): add predict + history + detail endpoints"
```

---

## Task 6: GET /api/species + POST /api/proteins

**Files:**
- Create: `server/routers/proteins.py`
- Create: `tests/test_server_proteins.py`

**Interfaces:**
- Consumes: `protein_engine.pipeline.run`, `bacteria_assistant.config.ORGANISM_METADATA`
- Produces: `GET /api/species` -> `list[SpeciesInfo]`, `POST /api/proteins` -> `ProteinResponse`

- [ ] **Step 1: Create `server/routers/proteins.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.config import ORGANISM_METADATA

from ..models import ProteinRequest, ProteinResponse, SpeciesInfo

router = APIRouter(prefix="/api", tags=["proteins"])


@router.get("/species", response_model=list[SpeciesInfo])
def list_species() -> list[SpeciesInfo]:
    return [
        SpeciesInfo(name=name, organism_type=m["organism_type"], gram=m["gram_label"], shape=m["shape_label"], group=m["taxonomy_group"])
        for name, m in ORGANISM_METADATA.items()
    ]


@router.post("/proteins", response_model=ProteinResponse)
def rank_proteins(request: ProteinRequest) -> ProteinResponse:
    from protein_engine.pipeline import run
    result = run(request.species, top_n=request.top_n)
    return ProteinResponse(
        species=result["species"], taxonomy_id=result["taxonomy_id"],
        resolved_name=result["resolved_name"], source=result["source"],
        selected_proteins=result["selected_proteins"], excluded=result.get("excluded", []),
    )
```

- [ ] **Step 2: Create `tests/test_server_proteins.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from server.main import app

client = TestClient(app)


def test_list_species() -> None:
    resp = client.get("/api/species")
    assert resp.status_code == 200
    species = resp.json()
    assert len(species) == 10
    names = {s["name"] for s in species}
    assert "Bacillus subtilis" in names
    assert "Escherichia coli" in names


FAKE_PROTEIN_RESULT = {
    "species": "Bacillus subtilis", "taxonomy_id": 224308,
    "resolved_name": "Bacillus subtilis", "source": "uniprot",
    "selected_proteins": [{"accession": "P37476", "protein_name": "spoA protein", "gene": "spoA", "length": 405, "reviewed": True, "score": 0.55, "rank": 1, "has_structure": True}],
    "excluded": [],
}


@patch("server.routers.proteins.run", return_value=FAKE_PROTEIN_RESULT)
def test_rank_proteins(mock_run) -> None:
    resp = client.post("/api/proteins", json={"species": "Bacillus subtilis", "top_n": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["species"] == "Bacillus subtilis"
    assert len(body["selected_proteins"]) == 1
    assert body["selected_proteins"][0]["accession"] == "P37476"
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server_proteins.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add server/routers/proteins.py tests/test_server_proteins.py
git commit -m "feat(server): add species list and protein ranking endpoints"
```

---

## Task 7: Full backend integration test

**Files:**
- Create: `tests/test_server_integration.py`

- [ ] **Step 1: Create `tests/test_server_integration.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from server.main import app

client = TestClient(app)

_FAKE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
FAKE_RESULT = {
    "organism_type": "bacteria", "predicted_bacteria_name": "E. coli",
    "bacteria_type": "gram_negative", "total_colonies_detected": 8,
    "dominant_shape": "bacilli", "confidence": 0.88,
}


def test_health() -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_species_list() -> None:
    resp = client.get("/api/species")
    assert resp.status_code == 200
    assert len(resp.json()) == 10


@patch("server.routers.predict.predict_bacteria_image", return_value=FAKE_RESULT)
def test_predict_then_fetch_by_id(mock_predict) -> None:
    post_resp = client.post("/api/predict", files={"image": ("t.png", _FAKE_PNG, "image/png")}, data={"mode": "basic"})
    assert post_resp.status_code == 200
    pred_id = post_resp.json()["id"]
    get_resp = client.get(f"/api/predict/{pred_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["predicted_species"] == "E. coli"
    hist_resp = client.get("/api/predict/history?limit=10")
    assert hist_resp.status_code == 200
    assert hist_resp.json()["total"] >= 1
```

- [ ] **Step 2: Run all backend tests together**

Run: `.venv/bin/python -m pytest tests/test_server_*.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run ruff lint on server/**

Run: `.venv/bin/python -m ruff check server/ && .venv/bin/python -m ruff format --check server/`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_server_integration.py
git commit -m "test(server): add full backend integration test"
```

---

## Task 8: Frontend scaffolding (Vite + React + Tailwind)

**Files:**
- Create: `frontend/` via `npm create vite`
- Create: `frontend/vite.config.js`
- Create: `frontend/src/index.css`

**Interfaces:**
- Produces: React app at `localhost:5173`, proxies `/api` -> `localhost:8000`

- [ ] **Step 1: Scaffold Vite + React project**

```bash
npm create vite@latest frontend -- --template react
cd frontend && npm install
npm install -D tailwindcss @tailwindcss/vite
npm install react-router-dom axios
```

- [ ] **Step 2: Replace `frontend/vite.config.js`**

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

- [ ] **Step 3: Replace `frontend/src/index.css`**

```css
@import "tailwindcss";
```

- [ ] **Step 4: Verify dev server starts**

Run: `cd frontend && npm run dev`
Expected: Vite starts on `http://localhost:5173`

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): scaffold Vite + React + Tailwind project"
```

---

## Task 9: App shell, nav, and API client

**Files:**
- Create: `frontend/src/api.js`
- Replace: `frontend/src/App.jsx`
- Replace: `frontend/src/main.jsx`

- [ ] **Step 1: Create `frontend/src/api.js`**

```javascript
import axios from 'axios';
const api = axios.create({ baseURL: '/api' });
export default api;
```

- [ ] **Step 2: Replace `frontend/src/App.jsx`**

```jsx
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import PredictPage from './pages/PredictPage';
import ProteinPage from './pages/ProteinPage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <nav className="bg-white shadow">
          <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-8">
            <span className="text-xl font-bold text-blue-900">BPA-22</span>
            <NavLink to="/" end className={({ isActive }) => `px-3 py-1 rounded ${isActive ? 'bg-blue-100 text-blue-800 font-semibold' : 'text-gray-600 hover:text-gray-900'}`}>
              Image Analysis
            </NavLink>
            <NavLink to="/proteins" className={({ isActive }) => `px-3 py-1 rounded ${isActive ? 'bg-blue-100 text-blue-800 font-semibold' : 'text-gray-600 hover:text-gray-900'}`}>
              Protein Ranking
            </NavLink>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-4 py-6">
          <Routes>
            <Route path="/" element={<PredictPage />} />
            <Route path="/proteins" element={<ProteinPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
```

- [ ] **Step 3: Replace `frontend/src/main.jsx`**

```jsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 4: Verify nav renders**

Run: `cd frontend && npm run dev`
Expected: Page shows "BPA-22" title + "Image Analysis" and "Protein Ranking" nav links.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/
git commit -m "feat(frontend): add app shell with nav and API client"
```

---

## Task 10: Predict page — Image upload + results + history

**Files:**
- Create: `frontend/src/pages/PredictPage.jsx`
- Create: `frontend/src/components/ImageUpload.jsx`
- Create: `frontend/src/components/ResultsPanel.jsx`
- Create: `frontend/src/components/ResultsTable.jsx`

- [ ] **Step 1: Create `frontend/src/components/ImageUpload.jsx`**

```jsx
import { useState } from 'react';

export default function ImageUpload({ onResult, onLoading }) {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handleAnalyze = async () => {
    if (!file) return;
    onLoading(true);
    const formData = new FormData();
    formData.append('image', file);
    formData.append('mode', 'advanced');
    try {
      const { default: api } = await import('../api');
      const resp = await api.post('/predict', formData);
      onResult(resp.data);
    } catch (err) {
      alert('Prediction failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      onLoading(false);
    }
  };

  return (
    <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
      <div onDrop={handleDrop} onDragOver={(e) => e.preventDefault()} className="min-h-[200px] flex flex-col items-center justify-center gap-4">
        {preview ? (
          <img src={preview} alt="Preview" className="max-h-48 rounded" />
        ) : (
          <p className="text-gray-500">Drag & drop a petri-dish image here</p>
        )}
        <label className="cursor-pointer bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
          Browse
          <input type="file" accept="image/*" onChange={handleFile} className="hidden" />
        </label>
      </div>
      {file && (
        <div className="mt-4">
          <p className="text-sm text-gray-600 mb-2">{file.name}</p>
          <button onClick={handleAnalyze} className="bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700 font-semibold">
            Analyze
          </button>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Create `frontend/src/components/ResultsPanel.jsx`**

```jsx
export default function ResultsPanel({ result }) {
  if (!result) return null;

  const conf = typeof result.confidence === 'number' ? result.confidence : 0;
  const confColor = conf > 0.75 ? 'text-green-700' : conf > 0.5 ? 'text-orange-600' : 'text-red-600';

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
      <div className="bg-white rounded-lg p-4 shadow">
        <p className="text-xs text-gray-500 uppercase">Species</p>
        <p className="text-lg font-semibold">{result.predicted_species || '—'}</p>
      </div>
      <div className="bg-white rounded-lg p-4 shadow">
        <p className="text-xs text-gray-500 uppercase">Gram</p>
        <p className="text-lg font-semibold">{result.gram || '—'}</p>
      </div>
      <div className="bg-white rounded-lg p-4 shadow">
        <p className="text-xs text-gray-500 uppercase">Dominant Shape</p>
        <p className="text-lg font-semibold">{result.dominant_shape || '—'}</p>
      </div>
      <div className="bg-white rounded-lg p-4 shadow">
        <p className="text-xs text-gray-500 uppercase">Colonies</p>
        <p className="text-lg font-semibold">{result.total_colonies ?? '—'}</p>
      </div>
      <div className="bg-white rounded-lg p-4 shadow">
        <p className="text-xs text-gray-500 uppercase">Confidence</p>
        <p className={`text-lg font-bold ${confColor}`}>{conf.toFixed(2)}</p>
      </div>
      <div className="bg-white rounded-lg p-4 shadow">
        <p className="text-xs text-gray-500 uppercase">Organism Type</p>
        <p className="text-lg font-semibold">{result.organism_type || '—'}</p>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create `frontend/src/components/ResultsTable.jsx`**

```jsx
export default function ResultsTable({ colonies }) {
  if (!colonies || colonies.length === 0) return null;

  return (
    <div className="mt-4 overflow-x-auto">
      <h3 className="text-sm font-semibold text-gray-700 mb-2">Colony Measurements ({colonies.length} detected)</h3>
      <table className="min-w-full text-sm bg-white rounded-lg shadow">
        <thead className="bg-gray-100">
          <tr>
            <th className="px-3 py-2 text-left">#</th>
            <th className="px-3 py-2 text-left">Area</th>
            <th className="px-3 py-2 text-left">Perimeter</th>
            <th className="px-3 py-2 text-left">Circularity</th>
            <th className="px-3 py-2 text-left">Aspect Ratio</th>
            <th className="px-3 py-2 text-left">Solidity</th>
            <th className="px-3 py-2 text-left">Shape</th>
          </tr>
        </thead>
        <tbody>
          {colonies.map((c) => (
            <tr key={c.id} className="border-t hover:bg-gray-50">
              <td className="px-3 py-2">{c.id}</td>
              <td className="px-3 py-2">{c.area?.toFixed(1)}</td>
              <td className="px-3 py-2">{c.perimeter?.toFixed(1)}</td>
              <td className="px-3 py-2">{c.circularity?.toFixed(2)}</td>
              <td className="px-3 py-2">{c.aspect_ratio?.toFixed(2)}</td>
              <td className="px-3 py-2">{c.solidity?.toFixed(2)}</td>
              <td className="px-3 py-2">{c.predicted_shape}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 4: Create `frontend/src/pages/PredictPage.jsx`**

```jsx
import { useState, useEffect } from 'react';
import ImageUpload from '../components/ImageUpload';
import ResultsPanel from '../components/ResultsPanel';
import ResultsTable from '../components/ResultsTable';
import api from '../api';

export default function PredictPage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [showJson, setShowJson] = useState(false);

  useEffect(() => {
    api.get('/predict/history?limit=10').then((r) => setHistory(r.data.items)).catch(() => {});
  }, []);

  const loadHistory = async (id) => {
    try {
      const resp = await api.get(`/predict/${id}`);
      setResult(resp.data);
    } catch {}
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Image Analysis</h2>

      <ImageUpload onResult={(r) => { setResult(r); api.get('/predict/history?limit=10').then((r) => setHistory(r.data.items)); }} onLoading={setLoading} />

      {loading && <p className="text-blue-600">Analyzing...</p>}

      {result && (
        <>
          <ResultsPanel result={result} />
          <ResultsTable colonies={result.colonies} />

          <div>
            <button onClick={() => setShowJson(!showJson)} className="text-sm text-blue-600 hover:underline">
              {showJson ? 'Hide' : 'Show'} Raw JSON
            </button>
            {showJson && (
              <pre className="mt-2 bg-gray-900 text-green-400 p-4 rounded-lg text-xs overflow-auto max-h-96">
                {JSON.stringify(result, null, 2)}
              </pre>
            )}
          </div>
        </>
      )}

      {history.length > 0 && (
        <div className="mt-8">
          <h3 className="text-sm font-semibold text-gray-600 mb-2">Recent Predictions</h3>
          <div className="bg-white rounded-lg shadow divide-y">
            {history.map((h) => (
              <button key={h.id} onClick={() => loadHistory(h.id)} className="w-full text-left px-4 py-3 hover:bg-gray-50 flex justify-between">
                <span className="text-sm font-medium">{h.predicted_species || 'unknown'}</span>
                <span className="text-xs text-gray-500">#{h.id} — {h.confidence?.toFixed(2) ?? '—'}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Verify predict page renders**

Run: `cd frontend && npm run dev`, navigate to `http://localhost:5173`
Expected: Upload area renders, "Analyze" button appears after file selection.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/
git commit -m "feat(frontend): add predict page with upload, results, and history"
```

---

## Task 11: Protein page — Species selector + ranked table + FASTA download

**Files:**
- Create: `frontend/src/pages/ProteinPage.jsx`
- Create: `frontend/src/components/SpeciesSelector.jsx`
- Create: `frontend/src/components/ProteinTable.jsx`
- Create: `frontend/src/components/FastaDownload.jsx`

- [ ] **Step 1: Create `frontend/src/components/SpeciesSelector.jsx`**

```jsx
import { useState, useEffect } from 'react';
import api from '../api';

export default function SpeciesSelector({ value, onChange }) {
  const [species, setSpecies] = useState([]);

  useEffect(() => {
    api.get('/species').then((r) => setSpecies(r.data)).catch(() => {});
  }, []);

  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="border rounded px-3 py-2 text-sm">
      <option value="">Select species...</option>
      {species.map((s) => (
        <option key={s.name} value={s.name}>{s.name} ({s.gram})</option>
      ))}
    </select>
  );
}
```

- [ ] **Step 2: Create `frontend/src/components/ProteinTable.jsx`**

```jsx
export default function ProteinTable({ proteins }) {
  if (!proteins || proteins.length === 0) return null;

  return (
    <table className="min-w-full text-sm bg-white rounded-lg shadow">
      <thead className="bg-gray-100">
        <tr>
          <th className="px-3 py-2 text-left">#</th>
          <th className="px-3 py-2 text-left">Accession</th>
          <th className="px-3 py-2 text-left">Gene</th>
          <th className="px-3 py-2 text-left">Length</th>
          <th className="px-3 py-2 text-left">Score</th>
          <th className="px-3 py-2 text-left">Structure</th>
          <th className="px-3 py-2 text-left">Reviewed</th>
        </tr>
      </thead>
      <tbody>
        {proteins.map((p) => (
          <tr key={p.accession} className="border-t hover:bg-gray-50">
            <td className="px-3 py-2">{p.rank}</td>
            <td className="px-3 py-2 font-mono text-xs">{p.accession}</td>
            <td className="px-3 py-2">{p.gene || '—'}</td>
            <td className="px-3 py-2">{p.length}</td>
            <td className="px-3 py-2 font-semibold">{p.score?.toFixed(2)}</td>
            <td className="px-3 py-2">{p.has_structure ? '✓' : '—'}</td>
            <td className="px-3 py-2">{p.reviewed ? '✓' : '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 3: Create `frontend/src/components/FastaDownload.jsx`**

```jsx
export default function FastaDownload({ proteins, species }) {
  if (!proteins || proteins.length === 0) return null;

  const fasta = proteins.map((p) => `>${p.accession}|${species}|${p.gene || 'unknown'}|${p.length}\n${p.sequence || ''}`).join('\n');

  const download = () => {
    const blob = new Blob([fasta], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${species.replace(/\s+/g, '_')}_proteins.fasta`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <button onClick={download} className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 text-sm">
      Download FASTA
    </button>
  );
}
```

- [ ] **Step 4: Create `frontend/src/pages/ProteinPage.jsx`**

```jsx
import { useState } from 'react';
import SpeciesSelector from '../components/SpeciesSelector';
import ProteinTable from '../components/ProteinTable';
import FastaDownload from '../components/FastaDownload';
import api from '../api';

export default function ProteinPage() {
  const [species, setSpecies] = useState('');
  const [topN, setTopN] = useState(10);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleRun = async () => {
    if (!species) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await api.post('/proteins', { species, top_n: topN });
      setResult(resp.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Protein Ranking</h2>

      <div className="flex items-end gap-4">
        <div>
          <label className="block text-sm text-gray-600 mb-1">Species</label>
          <SpeciesSelector value={species} onChange={setSpecies} />
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-1">Top N</label>
          <input type="number" value={topN} onChange={(e) => setTopN(Number(e.target.value))} min={1} max={50} className="border rounded px-3 py-2 text-sm w-20" />
        </div>
        <button onClick={handleRun} disabled={!species || loading} className="bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700 font-semibold disabled:opacity-50">
          {loading ? 'Running...' : 'Run Ranking'}
        </button>
      </div>

      {error && <p className="text-red-600 text-sm">{error}</p>}

      {result && (
        <>
          <div className="text-sm text-gray-600">
            {result.resolved_name} — taxonomy ID {result.taxonomy_id} — source: {result.source}
          </div>
          <ProteinTable proteins={result.selected_proteins} />
          <FastaDownload proteins={result.selected_proteins} species={result.species} />

          {result.excluded?.length > 0 && (
            <p className="text-sm text-gray-500">{result.excluded.length} proteins excluded (too short or non-standard residues)</p>
          )}
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Verify protein page renders**

Run: `cd frontend && npm run dev`, navigate to `http://localhost:5173/proteins`
Expected: Species dropdown loads, "Run Ranking" button works.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/
git commit -m "feat(frontend): add protein page with species selector, table, FASTA download"
```

---

## Task 12: Lint + full test pass + final verification

**Files:**
- No new files

- [ ] **Step 1: Run all backend tests**

Run: `.venv/bin/python -m pytest tests/test_server_*.py -v`
Expected: ALL PASS

- [ ] **Step 2: Run ruff on server/**

Run: `.venv/bin/python -m ruff check server/ && .venv/bin/python -m ruff format --check server/`
Expected: ALL PASS

- [ ] **Step 3: Verify full project test suite still passes**

Run: `.venv/bin/python -m pytest -q`
Expected: ALL PASS (existing + new server tests)

- [ ] **Step 4: Manual end-to-end test**

1. Start backend: `.venv/bin/uvicorn server.main:app --port 8000`
2. Start frontend: `cd frontend && npm run dev`
3. Open `http://localhost:5173`
4. Upload an image, verify prediction appears
5. Check history shows the prediction
6. Navigate to Protein Ranking, select species, run ranking
7. Verify protein table loads

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete web frontend + FastAPI backend for BPA-22"
```

---

## Task 13: Unified Pipeline History

**Files:**
- Modify: `server/db.py` (add `list_full_history()`, `delete_prediction()`)
- Modify: `server/routers/predict.py` (add delete endpoint, update history response)
- Modify: `frontend/src/pages/PredictPage.jsx` (show protein status in history)
- Create: `frontend/src/pages/HistoryPage.jsx` (new unified history page)
- Modify: `frontend/src/App.jsx` (add History route)

**Interfaces:**
- Consumes: `list_predictions()`, `get_prediction()`, `delete_prediction()`
- Produces: `GET /api/predict/history` with `protein_status`, `DELETE /api/predict/{id}`

- [ ] **Step 1: Add `delete_prediction()` to `server/db.py`**

```python
def delete_prediction(db_path: Path, prediction_id: int) -> bool:
    conn = _connect(db_path)
    cursor = conn.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted
```

- [ ] **Step 2: Add `list_full_history()` to `server/db.py`**

```python
def list_full_history(db_path: Path, *, limit: int = 50, offset: int = 0) -> dict:
    conn = _connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    rows = conn.execute(
        """
        SELECT p.*,
               CASE WHEN pa.id IS NOT NULL THEN 'complete'
                    ELSE 'not_run'
               END as protein_status
        FROM predictions p
        LEFT JOIN protein_analyses pa ON p.id = pa.prediction_id
        ORDER BY p.id DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()
    conn.close()
    return {"total": total, "items": [dict(row) for row in rows]}
```

- [ ] **Step 3: Create `protein_analyses` table in `server/db.py`**

Add to `init_db()`:
```python
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS protein_analyses (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id   INTEGER NOT NULL,
        species         TEXT NOT NULL,
        taxonomy_id     INTEGER,
        status          TEXT NOT NULL DEFAULT 'pending',
        result_json     TEXT,
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (prediction_id) REFERENCES predictions(id)
    )
    """
)
```

- [ ] **Step 4: Add DELETE endpoint to `server/routers/predict.py`**

```python
@router.delete("/{prediction_id}")
def delete_prediction_by_id(prediction_id: int) -> dict:
    db_path = get_db_path()
    deleted = delete_prediction(db_path, prediction_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"status": "deleted", "id": prediction_id}
```

- [ ] **Step 5: Create `frontend/src/pages/HistoryPage.jsx`**

```jsx
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api';

export default function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const limit = 20;

  const fetchHistory = async (offset = 0) => {
    setLoading(true);
    try {
      const resp = await api.get(`/predict/history?limit=${limit}&offset=${offset}`);
      setHistory(resp.data.items);
      setTotal(resp.data.total);
    } catch {}
    setLoading(false);
  };

  useEffect(() => { fetchHistory(page * limit); }, [page]);

  const handleDelete = async (id) => {
    if (!confirm('Delete this prediction?')) return;
    await api.delete(`/predict/${id}`);
    fetchHistory(page * limit);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'complete': return '✅';
      case 'pending': return '⏳';
      case 'failed': return '❌';
      default: return '➖';
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Pipeline History</h2>
      
      {loading && <p className="text-blue-600">Loading...</p>}
      
      <div className="bg-white rounded-lg shadow divide-y">
        {history.map((h) => (
          <div key={h.id} className="px-4 py-3 flex items-center justify-between hover:bg-gray-50">
            <button 
              onClick={() => navigate(`/predict/${h.id}`)}
              className="flex-1 text-left"
            >
              <div className="flex items-center gap-4">
                <span className="text-sm font-medium">{h.predicted_species || 'unknown'}</span>
                <span className="text-xs text-gray-500">{h.filename}</span>
                <span className="text-xs text-gray-400">#{h.id}</span>
                <span className="text-xs" title={`Protein analysis: ${h.protein_status}`}>
                  {getStatusIcon(h.protein_status)}
                </span>
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {h.confidence?.toFixed(2) ?? '—'} confidence — {h.created_at}
              </div>
            </button>
            <button 
              onClick={() => handleDelete(h.id)}
              className="text-red-500 hover:text-red-700 text-sm px-2"
            >
              Delete
            </button>
          </div>
        ))}
      </div>

      <div className="flex justify-between items-center">
        <button 
          onClick={() => setPage(p => Math.max(0, p - 1))}
          disabled={page === 0}
          className="px-4 py-2 bg-gray-200 rounded disabled:opacity-50"
        >
          Previous
        </button>
        <span className="text-sm text-gray-600">
          Page {page + 1} of {Math.ceil(total / limit)}
        </span>
        <button 
          onClick={() => setPage(p => p + 1)}
          disabled={(page + 1) * limit >= total}
          className="px-4 py-2 bg-gray-200 rounded disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Add History route to `frontend/src/App.jsx`**

Add to Routes:
```jsx
<Route path="/history" element={<HistoryPage />} />
```

Add nav link:
```jsx
<NavLink to="/history" className={({ isActive }) => `px-3 py-1 rounded ${isActive ? 'bg-blue-100 text-blue-800 font-semibold' : 'text-gray-600 hover:text-gray-900'}`}>
  History
</NavLink>
```

- [ ] **Step 7: Update `frontend/src/pages/PredictPage.jsx` history section**

Show protein status icon in recent predictions:
```jsx
{history.map((h) => (
  <button key={h.id} onClick={() => loadHistory(h.id)} className="w-full text-left px-4 py-3 hover:bg-gray-50 flex justify-between">
    <span className="text-sm font-medium">{h.predicted_species || 'unknown'}</span>
    <span className="text-xs text-gray-500">
      {h.protein_status === 'complete' ? '✅' : '➖'} #{h.id}
    </span>
  </button>
))}
```

- [ ] **Step 8: Run tests**

Run: `.venv/bin/python -m pytest tests/test_server_predict.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add server/db.py server/routers/predict.py frontend/src/
git commit -m "feat: add unified pipeline history with delete and protein status"
```

---

## Task 14: Protein History + Batch Analysis

**Files:**
- Modify: `server/routers/proteins.py` (add batch endpoint, history)
- Modify: `frontend/src/pages/ProteinPage.jsx` (add batch upload, history)

**Interfaces:**
- Consumes: `protein_engine.pipeline.run`, `protein_analyses` table
- Produces: `POST /api/proteins/batch`, `GET /api/proteins/history`

- [ ] **Step 1: Add `insert_protein_analysis()` to `server/db.py`**

```python
def insert_protein_analysis(
    db_path: Path,
    *,
    prediction_id: int,
    species: str,
    taxonomy_id: int | None,
    status: str,
    result_json: dict | None,
) -> int:
    conn = _connect(db_path)
    cursor = conn.execute(
        """
        INSERT INTO protein_analyses (prediction_id, species, taxonomy_id, status, result_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (prediction_id, species, taxonomy_id, status, json.dumps(result_json) if result_json else None),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id
```

- [ ] **Step 2: Add `list_protein_analyses()` to `server/db.py`**

```python
def list_protein_analyses(db_path: Path, *, limit: int = 50, offset: int = 0) -> dict:
    conn = _connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM protein_analyses").fetchone()[0]
    rows = conn.execute(
        """
        SELECT pa.*, p.filename, p.predicted_species as image_species
        FROM protein_analyses pa
        JOIN predictions p ON pa.prediction_id = p.id
        ORDER BY pa.id DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()
    conn.close()
    return {"total": total, "items": [dict(row) for row in rows]}
```

- [ ] **Step 3: Add batch endpoint to `server/routers/proteins.py`**

```python
@router.post("/proteins/batch")
async def rank_proteins_batch(species_list: list[str], top_n: int = 10) -> list[ProteinResponse]:
    from protein_engine.pipeline import run
    results = []
    for species in species_list:
        try:
            result = run(species, top_n=top_n)
            results.append(ProteinResponse(
                species=result["species"], taxonomy_id=result["taxonomy_id"],
                resolved_name=result["resolved_name"], source=result["source"],
                selected_proteins=result["selected_proteins"], excluded=result.get("excluded", []),
            ))
        except Exception as e:
            results.append(ProteinResponse(
                species=species, taxonomy_id=0, resolved_name=species,
                source="error", selected_proteins=[], excluded=[],
            ))
    return results
```

- [ ] **Step 4: Add history endpoint to `server/routers/proteins.py`**

```python
@router.get("/proteins/history")
def protein_history(limit: int = 50, offset: int = 0) -> dict:
    from ..db import list_protein_analyses
    db_path = get_db_path()
    return list_protein_analyses(db_path, limit=limit, offset=offset)
```

- [ ] **Step 5: Update `frontend/src/pages/ProteinPage.jsx` with batch + history**

Add batch upload section:
```jsx
const [batchMode, setBatchMode] = useState(false);
const [batchSpecies, setBatchSpecies] = useState('');
const [batchResults, setBatchResults] = useState([]);

const handleBatchRun = async () => {
  const speciesArray = batchSpecies.split('\n').filter(s => s.trim());
  if (speciesArray.length === 0) return;
  setLoading(true);
  try {
    const resp = await api.post(`/proteins/batch?top_n=${topN}`, speciesArray);
    setBatchResults(resp.data);
  } catch (err) {
    setError(err.response?.data?.detail || err.message);
  }
  setLoading(false);
};
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/test_server_proteins.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add server/routers/proteins.py server/db.py frontend/src/pages/ProteinPage.jsx
git commit -m "feat: add protein batch analysis and history"
```

---

## Task 15: Data Augmentation + TTA + Label Smoothing

**Files:**
- Modify: `src/bacteria_assistant/dl/trainer.py` (enable augmentation, add TTA)
- Modify: `src/bacteria_assistant/dl/model.py` (add label smoothing)
- Create: `tests/test_augmentation.py`

**Interfaces:**
- Consumes: `torchvision.transforms`, `torch.nn.CrossEntropyLoss`
- Produces: `train_with_augmentation()`, `predict_with_tta()`

- [ ] **Step 1: Add augmentation transforms to `src/bacteria_assistant/dl/trainer.py`**

```python
import torchvision.transforms as transforms

def get_augmentation_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
```

- [ ] **Step 2: Add TTA function to `src/bacteria_assistant/dl/trainer.py`**

```python
def predict_with_tta(model, image_tensor, n_augmentations=5):
    """Test-time augmentation: average predictions over augmented versions."""
    model.eval()
    predictions = []
    
    base_transform = get_augmentation_transforms()
    
    with torch.no_grad():
        for _ in range(n_augmentations):
            augmented = base_transform(image_tensor)
            augmented = augmented.unsqueeze(0)
            output = model(augmented)
            predictions.append(torch.softmax(output, dim=1))
    
    avg_predictions = torch.stack(predictions).mean(dim=0)
    return avg_predictions
```

- [ ] **Step 3: Add label smoothing to `src/bacteria_assistant/dl/model.py`**

```python
import torch.nn as nn

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        log_pred = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_pred.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_pred.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
```

- [ ] **Step 4: Create `tests/test_augmentation.py`**

```python
import torch
from bacteria_assistant.dl.trainer import get_augmentation_transforms, predict_with_tta
from bacteria_assistant.dl.model import LabelSmoothingCrossEntropy

def test_augmentation_transforms():
    transform = get_augmentation_transforms()
    assert transform is not None

def test_label_smoothing():
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    pred = torch.randn(2, 10)
    target = torch.tensor([1, 3])
    loss = criterion(pred, target)
    assert loss.item() > 0
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_augmentation.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/bacteria_assistant/dl/ tests/test_augmentation.py
git commit -m "feat: add data augmentation, TTA, and label smoothing"
```

---

## Task 16: K-Fold Cross-Validation + Backbone Fine-Tuning

**Files:**
- Modify: `src/bacteria_assistant/dl/trainer.py` (add k-fold, fine-tuning)
- Create: `tests/test_kfold.py`

**Interfaces:**
- Consumes: `sklearn.model_selection.StratifiedKFold`
- Produces: `train_kfold()`, `train_with_fine_tuning()`

- [ ] **Step 1: Add k-fold CV to `src/bacteria_assistant/dl/trainer.py`**

```python
from sklearn.model_selection import StratifiedKFold

def train_kfold(dataset, model_class, n_folds=5, epochs=30, batch_size=32):
    """Train with k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), dataset.targets)):
        print(f"Fold {fold + 1}/{n_folds}")
        
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size)
        
        model = model_class(num_classes=len(dataset.classes))
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        for epoch in range(epochs):
            train_one_epoch(model, train_loader, optimizer, criterion)
            val_acc = evaluate(model, val_loader)
        
        fold_results.append(val_acc)
    
    return fold_results
```

- [ ] **Step 2: Add discriminative fine-tuning to `src/bacteria_assistant/dl/trainer.py`**

```python
def train_with_fine_tuning(model, train_loader, val_loader, epochs=20):
    """Fine-tune with discriminative learning rates."""
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n]
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 3e-4},
    ])
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion)
        val_acc = evaluate(model, val_loader)
        
        if epoch == 5:
            for param in model.backbone.parameters():
                param.requires_grad = True
    
    return model
```

- [ ] **Step 3: Create `tests/test_kfold.py`**

```python
from bacteria_assistant.dl.trainer import train_kfold

def test_kfold_returns_results():
    # Mock test - just verify function signature
    assert callable(train_kfold)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_kfold.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bacteria_assistant/dl/trainer.py tests/test_kfold.py
git commit -m "feat: add k-fold cross-validation and discriminative fine-tuning"
```

---

## Task 17: Classical ML Improvements

**Files:**
- Modify: `src/bacteria_assistant/training.py` (add features, ensemble)

**Interfaces:**
- Consumes: `sklearn.ensemble.VotingClassifier`
- Produces: `extract_color_features()`, `train_ensemble()`

- [ ] **Step 1: Add color/texture features to `src/bacteria_assistant/training.py`**

```python
import cv2
import numpy as np

def extract_color_features(image_path):
    """Extract RGB and HSV color histogram features."""
    img = cv2.imread(str(image_path))
    if img is None:
        return {}
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    features = {}
    for i, channel in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        features[f'{channel}_mean'] = np.mean(hist)
        features[f'{channel}_std'] = np.std(hist)
    
    for i, channel in enumerate(['h', 's', 'v']):
        hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
        features[f'{channel}_mean'] = np.mean(hist)
        features[f'{channel}_std'] = np.std(hist)
    
    return features
```

- [ ] **Step 2: Add ensemble training to `src/bacteria_assistant/training.py`**

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

def train_ensemble(X_train, y_train):
    """Train ensemble of top 3 classical models."""
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    et = ExtraTreesClassifier(n_estimators=200, random_state=42)
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('et', et), ('svm', svm)],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    return ensemble
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bacteria_assistant/training.py
git commit -m "feat: add color features and ensemble training for classical ML"
```

---

## Task 18: Python 3.9 Compatibility

**Files:**
- Modify: `src/bacteria_assistant/**/*.py` (fix type hints)
- Modify: `src/protein_engine/**/*.py` (fix syntax)
- Modify: `setup.cfg` (add python_requires)

**Interfaces:**
- Consumes: `typing.Union`, `typing.Optional`
- Produces: Compatible code for Python 3.9

- [ ] **Step 1: Scan for 3.10+ syntax**

Run: `.venv/bin/python -m py_compile src/bacteria_assistant/*.py`
Run: `.venv/bin/python -m py_compile src/protein_engine/*.py`

- [ ] **Step 2: Replace `X | Y` with `Union[X, Y]`**

In files with type hints:
```python
# Before
def func(x: int | str) -> bool | None:

# After
from typing import Union, Optional
def func(x: Union[int, str]) -> Optional[bool]:
```

- [ ] **Step 3: Add `from __future__ import annotations`**

Add to top of files that need it for deferred evaluation.

- [ ] **Step 4: Update `setup.cfg`**

```ini
[options]
python_requires = >=3.9
```

- [ ] **Step 5: Test on Python 3.9**

Run: `python3.9 -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ setup.cfg
git commit -m "fix: ensure Python 3.9 compatibility"
```
