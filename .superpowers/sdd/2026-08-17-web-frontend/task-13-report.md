# Task 13: Unified Pipeline History — Report

## What was implemented

- `server/db.py`: Added `protein_analyses` table in `init_db()`, `list_full_history()` with LEFT JOIN for protein_status, `delete_prediction()` function
- `server/routers/predict.py`: Added DELETE `/api/predict/{prediction_id}` endpoint, switched history endpoint to use `list_full_history()`
- `server/models.py`: Added `protein_status` field to `HistoryItem`
- `frontend/src/pages/HistoryPage.jsx`: New page with paginated history, delete buttons, protein status icons, navigation to prediction detail
- `frontend/src/App.jsx`: Added `/history` route and nav link
- `frontend/src/pages/PredictPage.jsx`: Updated recent predictions to show protein status icons
- `tests/test_server_predict.py`: Added `init_db()` call to ensure DB schema is created before tests (pre-existing gap — `TestClient(app)` at module level doesn't trigger lifespan)

## What was tested

- `pytest tests/test_server_predict.py -v` — 4/4 passing
- Manual assertion test verifying `list_full_history` returns `protein_status` and `delete_prediction` works correctly

## Files changed

- `server/db.py` — added table, two new functions
- `server/routers/predict.py` — added DELETE endpoint, switched history to use list_full_history
- `server/models.py` — added protein_status to HistoryItem
- `frontend/src/pages/HistoryPage.jsx` — new file
- `frontend/src/App.jsx` — added route and nav link
- `frontend/src/pages/PredictPage.jsx` — protein status icon in recent list
- `tests/test_server_predict.py` — added init_db call

## Self-review findings

None. Implementation follows the task spec exactly. The only deviation is adding `init_db()` to the test module, which was a pre-existing gap in test setup (the lifespan handler wasn't being invoked by the module-level `TestClient`).

---

## Code review fix — 2026-08-17

**Commit:** `2c772df` on `feat/web-frontend`

### Fixes applied

1. **Silent error swallowing** (`frontend/src/pages/HistoryPage.jsx:19`): Added `error` state, reset it on each fetch, populated it from the catch block, and rendered a red error message in the JSX.

2. **Missing cascade delete** (`server/db.py:134`): Added `DELETE FROM protein_analyses WHERE prediction_id = ?` before the `predictions` delete so orphaned rows are cleaned up.

### Tests run

```
.venv/bin/python -m pytest tests/test_server_predict.py -v
```

**Result:** 4/4 passed (1.77s).

| Test | Status |
|------|--------|
| test_predict_returns_result | PASSED |
| test_predict_saves_to_history | PASSED |
| test_get_prediction_by_id | PASSED |
| test_get_prediction_404_for_nonexistent_id | PASSED |
