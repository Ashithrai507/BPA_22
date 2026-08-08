from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
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
            (species, entry["taxonomy_id"], entry["resolved_name"], datetime.now(UTC).isoformat()),
        )
        self._db.commit()
