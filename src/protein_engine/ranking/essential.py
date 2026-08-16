from __future__ import annotations

from pathlib import Path

from protein_engine.ranking._common import CuratedTable, load_curated_table

GENE_COLUMNS = ("gene", "gene_symbol", "symbol")
ACCESSION_COLUMNS = ("accession", "uniprot")


def load_essential_genes(path: Path) -> CuratedTable:
    return load_curated_table(path, "essential", GENE_COLUMNS, ACCESSION_COLUMNS)
