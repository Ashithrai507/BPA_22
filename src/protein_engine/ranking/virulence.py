from __future__ import annotations

from pathlib import Path

from protein_engine.ranking._common import CuratedTable, load_curated_table
from protein_engine.ranking.essential import ACCESSION_COLUMNS, GENE_COLUMNS


def load_virulence_genes(path: Path) -> CuratedTable:
    return load_curated_table(path, "virulence", GENE_COLUMNS, ACCESSION_COLUMNS)
