from __future__ import annotations

import json

import pytest


class FakeResponse:
    def __init__(self, text: str = "", status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


class FakeTransport:
    def __init__(self, responses: list[FakeResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def get(self, url: str, params: dict | None = None, headers: dict | None = None) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "headers": headers})
        if not self.responses:
            raise AssertionError(f"unexpected request: {url} {params}")
        return self.responses.pop(0)


def json_response(payload: object, status_code: int = 200) -> FakeResponse:
    return FakeResponse(json.dumps(payload), status_code=status_code)


@pytest.fixture
def fake_transport() -> FakeTransport:
    return FakeTransport()
