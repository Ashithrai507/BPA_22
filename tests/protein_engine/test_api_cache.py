from __future__ import annotations

import pytest
from conftest import FakeResponse, FakeTransport, json_response

from protein_engine.retrieval.api_cache import ApiCache, ApiError, CacheMissError


def _cache(path, transport, **kwargs):
    return ApiCache(path, transport=transport, rate_limit=0.0, backoff=[], **kwargs)


def test_fetch_caches_and_serves_from_cache(tmp_path) -> None:
    transport = FakeTransport([json_response({"a": 1})])
    cache = _cache(tmp_path / "http.db", transport)
    assert cache.fetch("http://x", params={"q": "1"}) == '{"a": 1}'
    assert cache.fetch("http://x", params={"q": "1"}) == '{"a": 1}'
    assert len(transport.calls) == 1


def test_fetch_retries_then_succeeds(tmp_path) -> None:
    transport = FakeTransport([FakeResponse("", status_code=503), FakeResponse("ok")])
    cache = ApiCache(tmp_path / "http.db", transport=transport, rate_limit=0.0, backoff=[0.0, 0.0])
    assert cache.fetch("http://x") == "ok"
    assert len(transport.calls) == 2


def test_fetch_raises_api_error_on_non_retryable_status(tmp_path) -> None:
    transport = FakeTransport([FakeResponse("", status_code=404)])
    cache = _cache(tmp_path / "http.db", transport)
    with pytest.raises(ApiError) as excinfo:
        cache.fetch("http://x")
    assert excinfo.value.status_code == 404


def test_fetch_offline_uses_cache_else_raises(tmp_path) -> None:
    transport = FakeTransport([json_response({"a": 1})])
    cache = _cache(tmp_path / "http.db", transport)
    cache.fetch("http://x")
    assert cache.fetch("http://x", allow_network=False) == '{"a": 1}'
    with pytest.raises(CacheMissError):
        cache.fetch("http://other", allow_network=False)


def test_fetch_allow_network_false_uses_constructor_default(tmp_path) -> None:
    transport = FakeTransport([json_response({"a": 1})])
    cache = _cache(tmp_path / "http.db", transport, allow_network=False)
    with pytest.raises(CacheMissError):
        cache.fetch("http://x")
    cache.set("http://x", None, '{"a": 1}')
    assert cache.fetch("http://x") == '{"a": 1}'
    assert len(transport.calls) == 0
