from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


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
