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


def list_full_history(db_path: Path, *, limit: int = 50, offset: int = 0) -> dict:
    conn = _connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    rows = conn.execute(
        """
        SELECT p.*,
               CASE WHEN latest_pa.id IS NOT NULL THEN 'complete'
                    ELSE 'not_run'
               END as protein_status
        FROM predictions p
        LEFT JOIN (
            SELECT prediction_id, MAX(id) AS id
            FROM protein_analyses
            GROUP BY prediction_id
        ) latest_pa ON p.id = latest_pa.prediction_id
        ORDER BY p.id DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()
    conn.close()
    return {"total": total, "items": [dict(row) for row in rows]}


def delete_prediction(db_path: Path, prediction_id: int) -> bool:
    conn = _connect(db_path)
    conn.execute("DELETE FROM protein_analyses WHERE prediction_id = ?", (prediction_id,))
    cursor = conn.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


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
