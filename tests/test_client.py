from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

from jainsons_llm_router import (
    BillingClass,
    ConfigurationError,
    LedgerUnavailable,
    RouteUnavailable,
    client,
)
from jainsons_llm_router.adapters import GenericHTTPAdapter
from jainsons_llm_router.models import CallerContext, LLMRequest, UseRoute
from jainsons_llm_router.policies.harness_derived import HarnessBackendPolicy


def _policy(name: str, kind: str, model: str | None, enabled: bool = True) -> HarnessBackendPolicy:
    return HarnessBackendPolicy(
        name=name,
        kind=kind,
        funding="paid_api" if kind == "API$$" else "free_service",
        billing_class=BillingClass.PAID if kind == "API$$" else BillingClass.FREE,
        model=model,
        model_source="test",
        automatic_enabled=enabled,
    )


def _set_http_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    policies = {
        "codex": _policy("codex", "sub", "gpt-5.6-terra"),
        "agy": _policy("agy", "sub-free", None),
        "claude": _policy("claude", "sub-anthropic", None),
        "omni": _policy("omni", "omni-free", "omni/free"),
        "paid": _policy("paid", "API$$", "paid/model"),
        "or-free-ling": _policy("or-free-ling", "or-free", "qwen/ling:free"),
        "or-free-alpha": _policy("or-free-alpha", "or-free", "meta/alpha:free"),
        "or-free-beta": _policy("or-free-beta", "or-free", "google/beta:free"),
    }
    chains = {
        "research": ("or-free-ling", "or-free-alpha", "or-free-beta"),
        "writing": ("or-free-ling", "or-free-alpha", "or-free-beta"),
        "code": ("codex", "agy", "claude", "omni", "paid", "or-free-ling"),
        "ui": ("codex", "agy", "claude", "omni", "paid", "or-free-ling"),
        "planning": (),
        "domain_ops": (),
        "legal_finance": (),
    }
    monkeypatch.setattr(client.harness_derived, "BACKENDS", policies)
    monkeypatch.setattr(client.harness_derived, "HTTP_CHAIN_BY_LANE", chains, raising=False)


def test_research_yields_live_or_free_candidates_in_harness_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_http_policy(monkeypatch)
    monkeypatch.setattr(client.free_check, "model_is_free", lambda model: True)
    candidates = client._lane_candidates("research")
    assert [candidate.provider for candidate in candidates] == ["or-free-ling", "or-free-alpha", "or-free-beta"]
    assert all(candidate.billing_class is BillingClass.FREE for candidate in candidates)
    assert all(candidate.timeout_seconds == 120.0 for candidate in candidates)


def test_client_never_converts_cli_or_paid_backends_to_http_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_http_policy(monkeypatch)
    monkeypatch.setattr(client.free_check, "model_is_free", lambda model: True)
    for lane in ("code", "ui"):
        candidates = client._lane_candidates(lane)
        assert [candidate.provider for candidate in candidates] == ["or-free-ling"]
        assert candidates[0].model != "gpt-5.6-terra"


def test_candidate_build_is_static_and_planning_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_http_policy(monkeypatch)
    monkeypatch.setattr(client.free_check, "model_is_free", lambda model: model != "meta/alpha:free")
    # Live eligibility is deliberately checked by the cached adapter at
    # dispatch, so a later catalog change can make this model eligible again.
    assert [candidate.provider for candidate in client._lane_candidates("research")] == ["or-free-ling", "or-free-alpha", "or-free-beta"]
    with pytest.raises(RouteUnavailable, match="no currently-free HTTP model under harness rules"):
        client._lane_candidates("planning")


def test_cached_router_rechecks_each_free_candidate_at_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached route must not preserve a model's former free status."""
    _set_http_policy(monkeypatch)
    client._ROUTERS.clear()
    sent_models: list[str] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"output":"ok","usage":{"prompt_tokens":1,"completion_tokens":1}}'

    def opener(request, timeout):
        sent_models.append(json.loads(request.data.decode("utf-8"))["model"])
        return _Response()

    def adapter_factory(**kwargs):
        return GenericHTTPAdapter(**kwargs, opener=opener)

    current = {"qwen/ling:free": True, "meta/alpha:free": True, "google/beta:free": True}
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "test-key")
    monkeypatch.setattr(client, "GenericHTTPAdapter", adapter_factory)
    monkeypatch.setattr(client.free_check, "model_is_free", lambda model: current.get(model, False))
    router = client._get_router("https://router.test/chat", "TEST_OPENROUTER_KEY")
    caller = CallerContext("test", "test", "research", "test-v1", "correlation")
    request = LLMRequest(input="hello")
    assert router.complete(request, selection=UseRoute("research"), caller=caller).model == "qwen/ling:free"
    current["qwen/ling:free"] = False
    assert router.complete(request, selection=UseRoute("research"), caller=caller).model == "meta/alpha:free"
    current["qwen/ling:free"] = True
    assert router.complete(request, selection=UseRoute("research"), caller=caller).model == "qwen/ling:free"
    assert sent_models == ["qwen/ling:free", "meta/alpha:free", "qwen/ling:free"]
    assert client._get_router("https://router.test/chat", "TEST_OPENROUTER_KEY") is router
    client._ROUTERS.clear()


def test_free_only_ledger_refuses_settlement() -> None:
    with pytest.raises(LedgerUnavailable, match="free-only"):
        client._FreeOnlyLedger().settle("reservation", actual_micro_usd=0)


def test_unknown_lane_raises_clear_error_without_building_router(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "_get_router", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("router should not be built")))
    with pytest.raises(ConfigurationError, match="unknown harness lane"):
        client.complete_text("hello", lane="does-not-exist")


def test_module_import_has_no_router_or_cli_side_effects() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    pythonpath = os.pathsep.join(part for part in (str(source_root), os.environ.get("PYTHONPATH", "")) if part)
    completed = subprocess.run([sys.executable, "-c", "import jainsons_llm_router.client as client; assert client._ROUTERS == {}; print('import-clean')"], check=True, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": pythonpath})
    assert completed.stdout.strip() == "import-clean"
    assert completed.stderr == ""
