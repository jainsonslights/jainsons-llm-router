from __future__ import annotations

from pathlib import Path

import pytest

from jainsons_llm_router import BillingClass, Candidate, ConfigurationError
from jainsons_llm_router.policies.harness_derived import (
    BACKENDS,
    FREE_CANDIDATE_BACKENDS_BY_LANE,
    GLM_AUTOMATIC_DISABLED,
    order_free_candidates,
)
from jainsons_llm_router.sync_from_harness import (
    HarnessSyncError,
    main,
    parse_harness_source,
    render_policy_module,
)

FIXTURE = Path(__file__).parent / "fixtures" / "harness_policy_snapshot.py"


def _free_candidate(backend: str) -> Candidate:
    return Candidate(
        provider=backend,
        model=f"local-{backend}-pin",
        adapter=backend,
        billing_class=BillingClass.FREE,
        provider_account_alias=f"{backend}-account",
        zero_marginal_cost=True,
    )


def test_fixture_generates_expected_static_policy() -> None:
    facts = parse_harness_source(FIXTURE.read_text(encoding="utf-8"), filename=str(FIXTURE))

    assert facts.default_lane == "research"
    assert facts.auto_disabled_backends == frozenset({"glm", "kimi"})
    assert facts.backends["codex"].model == "gpt-fixture-main"
    assert facts.backends["codex-sol"].model == "gpt-fixture-hard"
    assert facts.backends["glm"].model == "glm-fixture"
    assert facts.backends["glm"].automatic_enabled is False
    assert facts.backends["or-best"].billing_class == "paid"
    assert facts.backends["or-free-fixture"].model is None
    assert facts.backends["or-free-fixture"].model_source == (
        "CATALOG['or-free-fixture']['id']"
    )
    assert facts.free_candidate_backends_by_lane == {
        "research": ("agy", "codex"),
        "code": ("codex", "agy"),
        "image_gen": ("agy",),
        "planning": ("claude", "codex"),
    }

    rendered = render_policy_module(facts)
    assert "HARNESS_POLICY_SHA256" in rendered
    assert "model='gpt-fixture-main'" in rendered
    assert "'glm': HarnessBackendPolicy(" in rendered
    assert rendered == render_policy_module(facts)


def test_runtime_only_harness_changes_do_not_change_generated_policy() -> None:
    source = FIXTURE.read_text(encoding="utf-8")
    facts = parse_harness_source(source, filename=str(FIXTURE))
    with_runtime_change = source + "\ndef worker_health_retry_backoff():\n    return 'changed'\n"
    changed = parse_harness_source(with_runtime_change, filename=str(FIXTURE))

    assert render_policy_module(changed) == render_policy_module(facts)


def test_generated_helper_applies_order_and_requires_explicit_missing_acknowledgement() -> None:
    codex = _free_candidate("codex")
    agy = _free_candidate("agy")
    assert order_free_candidates("code", {"agy": agy, "codex": codex}) == (codex, agy)

    with pytest.raises(ConfigurationError, match="missing harness-derived"):
        order_free_candidates("code", {"codex": codex})
    assert order_free_candidates("code", {"codex": codex}, allow_missing={"agy"}) == (codex,)


def test_generated_live_snapshot_preserves_glm_off_and_lane_order() -> None:
    assert GLM_AUTOMATIC_DISABLED
    assert BACKENDS["glm"].automatic_enabled is False
    assert FREE_CANDIDATE_BACKENDS_BY_LANE["code"] == ("codex", "agy")


def test_check_mode_detects_drift_without_real_harness(tmp_path: Path, capsys) -> None:
    output = tmp_path / "harness_derived.py"
    common = ["--harness", str(FIXTURE), "--output", str(output)]

    assert main(common) == 0
    assert main([*common, "--check"]) == 0
    output.write_text(output.read_text(encoding="utf-8") + "# stale\n", encoding="utf-8")
    assert main([*common, "--check"]) == 1
    assert "drift detected" in capsys.readouterr().err


def test_missing_required_symbol_fails_loudly() -> None:
    source = FIXTURE.read_text(encoding="utf-8").replace(
        'AUTO_DISABLED_BACKENDS = {"glm", "kimi"}',
        'RENAMED_DISABLED_BACKENDS = {"glm", "kimi"}',
    )
    with pytest.raises(HarnessSyncError, match="AUTO_DISABLED_BACKENDS.*missing"):
        parse_harness_source(source, filename="refactored_harness.py")


def test_missing_referenced_model_symbol_fails_loudly() -> None:
    source = FIXTURE.read_text(encoding="utf-8").replace(
        'CODEX_MODEL = os.environ.get("HARNESS_CODEX_MODEL", "gpt-fixture-main")',
        'RENAMED_CODEX_MODEL = os.environ.get("HARNESS_CODEX_MODEL", "gpt-fixture-main")',
    )
    with pytest.raises(HarnessSyncError, match="CODEX_MODEL.*not assigned"):
        parse_harness_source(source, filename="missing_model_symbol_harness.py")


def test_enabling_glm_requires_human_router_review() -> None:
    source = FIXTURE.read_text(encoding="utf-8").replace(
        'AUTO_DISABLED_BACKENDS = {"glm", "kimi"}',
        'AUTO_DISABLED_BACKENDS = {"kimi"}',
    )
    with pytest.raises(HarnessSyncError, match="enables GLM"):
        parse_harness_source(source, filename="glm_enabled_harness.py")
