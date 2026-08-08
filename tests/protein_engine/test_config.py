from __future__ import annotations

from protein_engine.config import (
    DEFAULT_TOP_N,
    MIN_SEQUENCE_LENGTH,
    PAGE_SIZE,
    PRESELECT_LIMIT,
    RANKING_WEIGHTS,
    RATE_LIMIT_DEFAULT,
    RETRY_BACKOFF,
    REVIEWED_MIN_COUNT,
    UNIPROT_FIELDS,
    UNIPROT_SEARCH,
)


def test_config_constants() -> None:
    assert UNIPROT_SEARCH == "https://rest.uniprot.org/uniprotkb/search"
    assert REVIEWED_MIN_COUNT == 10
    assert MIN_SEQUENCE_LENGTH == 20
    assert PAGE_SIZE == 500
    assert DEFAULT_TOP_N == 10
    assert PRESELECT_LIMIT == 50
    assert RATE_LIMIT_DEFAULT == 3.0
    assert RETRY_BACKOFF == (0.5, 1.0, 2.0)
    assert set(RANKING_WEIGHTS) == {"essential", "virulence", "resistance", "evidence", "structure"}
    assert "sequence" in UNIPROT_FIELDS
