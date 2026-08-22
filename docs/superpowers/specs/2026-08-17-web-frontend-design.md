# Design: Web Frontend + FastAPI Backend for BPA-22

**Date:** 2026-08-17
**Status:** Draft — pending user review

## 1. Summary

Replace the PyQt5 desktop GUI with a web-based frontend + REST API backend.
Both Phase 1 (image analysis) and Phase 2 (protein ranking) are exposed.
A SQLite audit log stores every prediction for history/review.
Built and tested locally first, then containerized for server deployment.

## 2. Architecture

Three-component system, developed locally without Docker:

```
localhost:5173 (React/Vite dev server)
       │  /api proxy
       ▼
localhost:8000 (FastAPI/uvicorn)
       │  imports
       ▼
src/bacteria_assistant/ + src/protein_engine/  (existing, unchanged)
       │
       ▼
data/bpa.db  (SQLite — predictions audit log)
```

### File structure

```
BPA_22/
├── src/                        # existing (unchanged)
│   ├── bacteria_assistant/
│   └── protein_engine/
├── server/                     # NEW — FastAPI backend
│   ├── main.py                 # app entry, startup, CORS
│   ├── dependencies.py         # model loading (once at startup)
│   ├── db.py                   # SQLite init + helpers
│   ├── models.py               # Pydantic request/response schemas
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── predict.py          # POST /api/predict, GET /api/predict/history, GET /api/predict/{id}
│   │   └── proteins.py         # POST /api/proteins, GET /api/species
│   └── requirements.txt        # fastapi, uvicorn, python-multipart
├── frontend/                   # NEW — React app
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js          # dev proxy /api → localhost:8000
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx             # React Router
│   │   ├── api.js              # axios instance (baseURL: /api)
│   │   ├── pages/
│   │   │   ├── PredictPage.jsx
│   │   │   └── ProteinPage.jsx
│   │   └── components/
│   │       ├── ImageUpload.jsx
│   │       ├── ResultsPanel.jsx
│   │       ├── ResultsTable.jsx
│   │       ├── ProteinTable.jsx
│   │       └── FastaDownload.jsx
│   └── public/
├── data/
│   └── bpa.db                  # SQLite (created at runtime)
└── server/requirements.txt     # fastapi, uvicorn, python-multipart
```

## 3. Backend API

Base URL: `http://localhost:8000/api`

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/predict` | Upload image → full prediction (basic or advanced mode) |
| GET | `/api/predict/history` | List past predictions (newest first, paginated) |
| GET | `/api/predict/{id}` | Fetch a single past prediction by ID |
| POST | `/api/proteins` | Species → ranked protein list |
| GET | `/api/species` | List 10 supported species with metadata |

### POST /api/predict

**Request:** `multipart/form-data`
- `image`: file (PNG/JPG/BMP/TIFF/WEBP)
- `mode`: `"basic"` or `"advanced"` (default `"advanced"`)

**Response (advanced):**
```json
{
  "id": 42,
  "created_at": "2026-08-17T10:30:00Z",
  "organism_type": "bacteria",
  "predicted_species": "Bacillus subtilis",
  "gram": "gram_positive",
  "total_colonies": 13,
  "dominant_shape": "bacilli",
  "confidence": 1.0,
  "colonies": [
    {
      "id": 1, "area": 109.5, "perimeter": 45.69,
      "circularity": 0.65, "aspect_ratio": 1.42,
      "solidity": 0.89, "equivalent_diameter": 11.8,
      "mean_intensity": 132.4, "predicted_shape": "bacilli"
    }
  ],
  "morphology": {
    "dominant_shape": "bacilli",
    "distribution": "clustered",
    "confidence": 0.87
  }
}
```

Basic mode: same top-level fields minus `colonies` and `morphology`.

Side effect: saves to `predictions` table (audit log).

### GET /api/predict/history

**Query params:** `?limit=50&offset=0`

**Response:**
```json
{
  "total": 120,
  "items": [
    {
      "id": 42,
      "created_at": "2026-08-17T10:30:00Z",
      "filename": "Bacillus_subtilis_gram_stain_1.png",
      "predicted_species": "Bacillus subtilis",
      "confidence": 1.0
    }
  ]
}
```

### GET /api/predict/{id}

Full prediction by ID, same shape as POST response.

### POST /api/proteins

**Request (JSON):**
```json
{
  "species": "Bacillus subtilis",
  "top_n": 10
}
```

**Response:**
```json
{
  "species": "Bacillus subtilis",
  "taxonomy_id": 224308,
  "resolved_name": "Bacillus subtilis",
  "source": "uniprot",
  "selected_proteins": [
    {
      "accession": "P37476",
      "protein_name": "Stage II sporulation protein A",
      "gene": "spoA",
      "length": 405,
      "reviewed": true,
      "score": 0.55,
      "rank": 1,
      "has_structure": true,
      "score_breakdown": {
        "essential": 1.0,
        "virulence": 0.0,
        "resistance": 0.0,
        "evidence": 1.0,
        "structure": 1.0
      }
    }
  ],
  "excluded": []
}
```

No DB write — protein results are transient (network calls already cached by `protein_engine`).

### GET /api/species

```json
[
  {
    "name": "Bacillus subtilis",
    "organism_type": "bacteria",
    "gram": "gram_positive",
    "shape": "bacilli",
    "group": "gram_positive_bacilli"
  }
]
```

## 4. Database Schema

Single SQLite file: `data/bpa.db`

### `predictions` table

```sql
CREATE TABLE predictions (
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
);
```

- `image_bytes`: stored image for history replay/download (~200KB per 640x640 PNG).
- `result_json`: full prediction payload as JSON text (colonies, morphology, etc.).

No protein results table — `protein_engine` caches network calls in its own SQLite DB.

## 5. Data Flow

### Image Analysis

```
User drops image → React POST /api/predict → FastAPI
  → predict_bacteria_image(temp_path, mode)
  → INSERT INTO predictions
  → return JSON + id
  → React renders: ResultsPanel + ResultsTable + HistorySidebar
```

### Protein Ranking

```
User picks species → React POST /api/proteins → FastAPI
  → pipeline.run(species, top_n)
  → return JSON (no DB write)
  → React renders: ProteinTable + FastaDownload
```

### Model Loading

Both models (sklearn joblib + DL embedding .pt) loaded once at FastAPI startup,
held in memory. `dependencies.py` provides `get_models()` returning the cached dict.

## 6. Frontend Pages

### `/predict` — Image Analysis

- `ImageUpload.jsx`: drag-drop zone + file picker, shows preview thumbnail
- `ResultsPanel.jsx`: species, gram, shape, confidence, colony count
- `ResultsTable.jsx`: colony measurements table (sortable columns)
- `HistorySidebar.jsx`: recent predictions, click to reload
- Expandable raw JSON view (same as PyQt5 "Show Details")

### `/proteins` — Protein Ranking

- `SpeciesSelector.jsx`: autocomplete dropdown (from `GET /api/species`)
- `ProteinTable.jsx`: ranked list, sortable, expandable score breakdown per row
- `FastaDownload.jsx`: download `.fasta` file from results

### Styling

Tailwind CSS. Clean, minimal, responsive. No retro theme.

## 7. Local Development

```bash
# Terminal 1 — backend
pip install -r server/requirements.txt
uvicorn server.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend && npm install && npm run dev
```

Vite dev server proxies `/api` → `localhost:8000`.

## 8. Server Deployment (later phase)

Docker Compose with 3 services: frontend (nginx), backend (FastAPI), SQLite (volume).
GPU passthrough via NVIDIA Container Toolkit for CUDA inference.
Server: Intel i5, 8GB RAM, NVIDIA MTX 330.

## 9. Non-goals (this phase)

- User authentication / access control
- Image annotation / colony highlighting overlay
- Mobile-responsive layout
- Server deployment (local-first, containerize after approval)
