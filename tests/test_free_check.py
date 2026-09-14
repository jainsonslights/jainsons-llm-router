from __future__ import annotations

import json

import pytest

from jainsons_llm_router import free_check


class _Response:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_free_check_requires_free_suffix_and_fetches_fresh_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    def fake_urlopen(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response({"data": [{"id": "vendor/model:free", "pricing": {"prompt": "0", "completion": 0}}]})
    monkeypatch.setattr(free_check, "urlopen", fake_urlopen)
    assert not free_check.model_is_free("vendor/model")
    assert free_check.model_is_free("vendor/model:free")
    assert free_check.model_is_free("vendor/model:free")
    assert calls == 2


@pytest.mark.parametrize("payload", [
    {"data": [{"id": "vendor/model:free", "pricing": {"prompt": "0.1", "completion": "0"}}]},
    {"data": [
        {"id": "vendor/model:free", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "vendor/model:free", "pricing": {"prompt": "0", "completion": "0"}},
    ]},
    {"not_data": []},
])
def test_free_check_fails_closed_for_nonzero_duplicate_and_malformed(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    monkeypatch.setattr(free_check, "urlopen", lambda *args, **kwargs: _Response(payload))
    assert not free_check.model_is_free("vendor/model:free")


def test_free_check_fails_closed_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(free_check, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timeout")))
    assert not free_check.model_is_free("vendor/model:free")
