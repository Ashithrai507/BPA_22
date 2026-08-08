from __future__ import annotations

from protein_engine.taxonomy.cache import TaxonomyCache


def test_round_trip(tmp_path) -> None:
    cache = TaxonomyCache(tmp_path / "taxonomy.db")
    assert cache.get("bacillus subtilis") is None
    cache.set("bacillus subtilis", {"taxonomy_id": 224308, "resolved_name": "Bacillus subtilis subsp. subtilis"})
    assert cache.get("bacillus subtilis") == (
        {"taxonomy_id": 224308, "resolved_name": "Bacillus subtilis subsp. subtilis"}
    )


def test_set_overwrites(tmp_path) -> None:
    cache = TaxonomyCache(tmp_path / "taxonomy.db")
    cache.set("ecoli", {"taxonomy_id": 1, "resolved_name": "x"})
    cache.set("ecoli", {"taxonomy_id": 2, "resolved_name": "y"})
    assert cache.get("ecoli") == {"taxonomy_id": 2, "resolved_name": "y"}


def test_persists_across_instances(tmp_path) -> None:
    path = tmp_path / "taxonomy.db"
    TaxonomyCache(path).set("ecoli", {"taxonomy_id": 562, "resolved_name": "Escherichia coli"})
    assert TaxonomyCache(path).get("ecoli") == {"taxonomy_id": 562, "resolved_name": "Escherichia coli"}
