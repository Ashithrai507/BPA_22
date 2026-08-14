from __future__ import annotations

import json
import re
from datetime import UTC, datetime
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
        "query_timestamp": datetime.now(UTC).isoformat(),
        "selected_proteins": proteins,
        "excluded": excluded,
    }
    (out / "proteins.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "sequences.fasta").write_text(to_fasta(proteins, resolved_name))
    return out
