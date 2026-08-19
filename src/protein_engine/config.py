from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True)
class Paths:
    cache_dir: Path = _env_path("PROTEIN_CACHE_DIR", PROJECT_ROOT / "cache")
    output_dir: Path = _env_path("PROTEIN_OUTPUT_DIR", PROJECT_ROOT / "output")
    reference_dir: Path = _env_path("PROTEIN_REFERENCE_DIR", PROJECT_ROOT / "reference_data")


PATHS = Paths()

UNIPROT_BASE = "https://rest.uniprot.org"
UNIPROT_SEARCH = f"{UNIPROT_BASE}/uniprotkb/search"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_ESEARCH = f"{NCBI_BASE}/esearch.fcgi"
NCBI_ESUMMARY = f"{NCBI_BASE}/esummary.fcgi"
NCBI_EFETCH = f"{NCBI_BASE}/efetch.fcgi"
PDB_UNIPROT = "https://data.rcsb.org/rest/v1/core/uniprot"

UNIPROT_FIELDS = (
    "accession,reviewed,protein_name,gene_names,length,sequence,"
    "cc_function,cc_subcellular_location,xref_pdb"
)
REVIEWED_MIN_COUNT = 10
MIN_SEQUENCE_LENGTH = 20
PAGE_SIZE = 500
DEFAULT_TOP_N = 10
PRESELECT_LIMIT = 50

RANKING_WEIGHTS = {
    "essential": 0.30,
    "virulence": 0.25,
    "resistance": 0.20,
    "evidence": 0.15,
    "structure": 0.10,
}
RATE_LIMIT_DEFAULT = 3.0
RETRY_BACKOFF = (0.5, 1.0, 2.0)
