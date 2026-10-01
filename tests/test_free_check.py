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


def test_free_check_uses_live_pricing_not_suffix_and_fetches_fresh_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    def fake_urlopen(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response({"data": [
            {"id": "vendor/model:free", "pricing": {"prompt": "0", "completion": 0}},
            {"id": "stealth/bunny", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "vendor/paid", "pricing": {"prompt": "0.000001", "completion": "0"}},
        ]})
    monkeypatch.setattr(free_check, "urlopen", fake_urlopen)
    assert free_check.model_is_free("vendor/model:free")
    assert free_check.model_is_free("vendor/model:free")
    # Stealth models are free without the suffix; the openrouter/ prefix is dropped as in the harness.
    assert free_check.model_is_free("stealth/bunny")
    assert free_check.model_is_free("openrouter/stealth/bunny")
    assert not free_check.model_is_free("vendor/paid")
    assert not free_check.model_is_free("vendor/missing")
    assert calls == 6


@pytest.mark.parametrize("model_id", ["", ":free", "openrouter/", "openrouter/:free", None, 7])
def test_free_check_rejects_empty_or_bare_ids_without_network(monkeypatch: pytest.MonkeyPatch, model_id: object) -> None:
    def fail_urlopen(*args, **kwargs):
        raise AssertionError("no catalog fetch for a malformed id")
    monkeypatch.setattr(free_check, "urlopen", fail_urlopen)
    assert not free_check.model_is_free(model_id)  # type: ignore[arg-type]


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


def test_free_check_ignores_unrelated_malformed_catalog_entries_like_the_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"data": [None, "junk", {"id": "stealth/bunny", "pricing": {"prompt": "0", "completion": "0"}}]}
    monkeypatch.setattr(free_check, "urlopen", lambda *a, **k: _Response(payload))
    assert free_check.model_is_free("stealth/bunny")
