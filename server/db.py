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
            filename,
            image_bytes,
            mode,
            organism_type,
            predicted_species,
            gram,
            total_colonies,
            dominant_shape,
            confidence,
            json.dumps(result_json),
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
