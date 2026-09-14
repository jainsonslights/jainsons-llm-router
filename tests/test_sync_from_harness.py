from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from jainsons_llm_router import BillingClass, Candidate
from jainsons_llm_router import sync_from_harness as sync_module
from jainsons_llm_router.policies import harness_derived


def fixture_export() -> dict[str, object]:
    free = {
        "or-free-ling": "qwen/qwen3-coder:free",
        "or-free-alpha": "meta/alpha:free",
        "or-free-beta": "google/beta:free",
    }
    backends: dict[str, dict[str, object]] = {
        name: {"kind": "or-free", "model": model, "auto_disabled": False, "http_openrouter": True}
        for name, model in free.items()
    }
    backends.update({
        "codex": {"kind": "sub", "model": "gpt-5.6-terra", "auto_disabled": False, "http_openrouter": False},
        "agy": {"kind": "sub-free", "model": None, "auto_disabled": False, "http_openrouter": False},
        "claude": {"kind": "sub-anthropic", "model": None, "auto_disabled": False, "http_openrouter": False},
        "glm": {"kind": "sub-glm", "model": "glm-5.2", "auto_disabled": True, "http_openrouter": False},
        "omni-fast": {"kind": "omni-free", "model": "omni-free", "auto_disabled": False, "http_openrouter": False},
        "or-best": {"kind": "API$$", "model": "paid/model", "auto_disabled": False, "http_openrouter": False},
    })
    lanes: dict[str, dict[str, object]] = {}
    for lane in ("research", "writing"):
        lanes[lane] = {"primary": "codex", "fallback": "agy", "escalation_chain": ["codex", "agy", *free], "http_chain": list(free)}
    for lane in ("code", "ui"):
        lanes[lane] = {"primary": "codex", "fallback": "agy", "escalation_chain": ["codex", "or-free-ling"], "http_chain": ["or-free-ling"]}
    for lane in ("planning", "domain_ops", "legal_finance"):
        lanes[lane] = {"primary": "codex", "fallback": "agy", "escalation_chain": ["codex", "agy"], "http_chain": []}
    return {"schema": 1, "default_lane": "research", "openrouter_free_backends": list(free), "free_check": {"function": "openrouter_free_catalog._openrouter_model_is_free", "source_sha256": "fixture-free-check-hash"}, "backends": backends, "lanes": lanes}


def _mock_export(monkeypatch: pytest.MonkeyPatch, export: object, returncode: int = 0) -> None:
    monkeypatch.setattr(sync_module.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode, json.dumps(export), "failed export"))


def test_sync_renders_exported_models_http_chains_and_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_export(monkeypatch, fixture_export())
    monkeypatch.setattr(sync_module, "PORTED_FROM_HARNESS_SHA256", "fixture-free-check-hash")
    output = tmp_path / "harness_derived.py"
    assert sync_module.main(["--harness", "fake-harness.py", "--output", str(output)]) == 0
    rendered = output.read_text(encoding="utf-8")
    assert "HARNESS_FREE_CHECK_SHA256 = 'fixture-free-check-hash'" in rendered
    assert "OPENROUTER_FREE_BACKENDS = ('or-free-ling', 'or-free-alpha', 'or-free-beta')" in rendered
    assert "'research': ('or-free-ling', 'or-free-alpha', 'or-free-beta')" in rendered
    assert "model='gpt-5.6-terra'" in rendered
    assert "HTTP_CHAIN_BY_LANE" in rendered
    assert "from .. import free_check" in rendered
    assert sync_module.main(["--harness", "fake-harness.py", "--output", str(output), "--check"]) == 0


@pytest.mark.parametrize("export", ["not-json", {"schema": 2}, {**fixture_export(), "backends": {"or-free-ling": {"kind": "or-free", "model": None, "auto_disabled": False, "http_openrouter": True}}}])
def test_bad_export_exits_two_without_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, export: object) -> None:
    output = tmp_path / "harness_derived.py"
    if export == "not-json":
        monkeypatch.setattr(sync_module.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "{bad", ""))
    else:
        _mock_export(monkeypatch, export)
    assert sync_module.main(["--harness", "fake-harness.py", "--output", str(output)]) == 2
    assert not output.exists()


def test_failed_export_and_hash_mismatch_write_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "harness_derived.py"
    _mock_export(monkeypatch, fixture_export(), returncode=1)
    assert sync_module.main(["--harness", "fake", "--output", str(output)]) == 2
    assert not output.exists()
    _mock_export(monkeypatch, fixture_export())
    assert sync_module.main(["--harness", "fake", "--output", str(output)]) == 2
    assert "harness free-check changed; re-port router free_check.py" in capsys.readouterr().err
    assert not output.exists()


def test_accept_changed_hash_prints_only_and_does_not_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _mock_export(monkeypatch, fixture_export())
    output = tmp_path / "harness_derived.py"
    assert sync_module.main(["--harness", "fake", "--output", str(output), "--accept-free-check-hash"]) == 0
    assert capsys.readouterr().out.strip() == "fixture-free-check-hash"
    assert not output.exists()


def test_forbidden_paid_or_claude_chain_is_rejected() -> None:
    export = fixture_export()
    lanes = export["lanes"]
    assert isinstance(lanes, dict)
    lane = lanes["research"]
    assert isinstance(lane, dict)
    lane["escalation_chain"] = ["claude"]
    with pytest.raises(sync_module.HarnessSyncError, match="forbidden Claude/paid"):
        sync_module.parse_routing_export(export)


def test_automatic_excludes_paid_api_and_anthropic_even_without_export_opt_out() -> None:
    facts = sync_module.parse_routing_export(fixture_export())
    assert not facts.backends["claude"].automatic_enabled
    assert not facts.backends["or-best"].automatic_enabled


def test_openrouter_prefixed_or_free_model_is_rejected_without_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    export = fixture_export()
    backends = export["backends"]
    assert isinstance(backends, dict)
    backends["or-free-ling"]["model"] = "openrouter/qwen/qwen3-coder:free"
    _mock_export(monkeypatch, export)
    monkeypatch.setattr(sync_module, "PORTED_FROM_HARNESS_SHA256", "fixture-free-check-hash")
    output = tmp_path / "harness_derived.py"
    output.write_text("unchanged", encoding="utf-8")
    assert sync_module.main(["--harness", "fake-harness.py", "--output", str(output)]) == 2
    assert output.read_text(encoding="utf-8") == "unchanged"


def test_generated_free_order_excludes_cli_and_non_free_models(monkeypatch: pytest.MonkeyPatch) -> None:
    model = harness_derived.BACKENDS["or-free-ling"].model
    assert model is not None
    monkeypatch.setattr(harness_derived.free_check, "model_is_free", lambda value: value == model)
    order = harness_derived.free_candidate_backend_order("research")
    assert order == ("or-free-ling",)
    candidates = {
        "codex": Candidate("codex", "gpt-5.6-terra", "test", BillingClass.FREE, "codex", zero_marginal_cost=True),
        "or-free-ling": Candidate("or-free-ling", model, "test", BillingClass.FREE, "or-free-ling", zero_marginal_cost=True),
    }
    assert harness_derived.order_free_candidates("research", candidates) == (candidates["or-free-ling"],)
