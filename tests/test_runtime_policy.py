from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from jainsons_llm_router import RouteUnavailable, app_policy_status, complete_chat, set_alert_sink
from jainsons_llm_router import runtime_policy
from jainsons_llm_router.runtime_policy import RuntimePolicy, canonical_app_policy_sha256, validate_app_policy


def test_real_harness_export_fixture_has_the_canonical_app_policy_hash() -> None:
    fixture = Path(__file__).parent / "fixtures" / "promoted_app_policy.json"
    policy = json.loads(fixture.read_text(encoding="utf-8"))
    assert policy["app_policy_sha256"] == canonical_app_policy_sha256(
        policy["app_backends"], policy["app_lanes"]
    )
    assert RuntimePolicy(fixture).load().sha256 == policy["app_policy_sha256"]


def test_canonical_hash_uses_utf8_without_ascii_escaping() -> None:
    backends = {"b": {"label": "café"}}
    lanes = {"lane": {"backend_chain": ["b"]}}
    expected = hashlib.sha256(
        json.dumps(
            {"app_backends": backends, "app_lanes": lanes},
            ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert canonical_app_policy_sha256(backends, lanes) == expected


def promoted_policy() -> dict[str, object]:
    backends = {
        "deepseek-direct-flash": {
            "kind": "app-metered",
            "model": "deepseek-flash",
            "transport": {
                "endpoint": "https://api.deepseek.com/chat/completions",
                "allowed_host": "api.deepseek.com",
                "key_env": "DEEPSEEK_API_KEY",
                "dialect": "openai-chat-completions",
                "payload_defaults": {"thinking": {"type": "disabled"}},
            },
            "price_card": {"version": "deepseek-v4.1-flash-peak-2026-09", "input_usd_per_million": 0.30, "output_usd_per_million": 1.20},
        },
        "glm-openrouter-flash": {
            "kind": "app-metered",
            "model": "z-ai/glm-5.3-flash",
            "transport": {
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "allowed_host": "openrouter.ai",
                "key_env": "OPENROUTER_API_KEY",
                "dialect": "openai-chat-completions",
                "payload_defaults": {"provider": {"data_collection": "deny"}},
            },
            "price_card": {"version": "glm-5.3-flash-2026-09", "input": 0.075, "output": 0.25},
        },
    }
    lanes = {
        "chat_fast": {
            "purpose": "live customer text",
            "backend_chain": ["deepseek-direct-flash", "glm-openrouter-flash"],
            "max_output_tokens": 1300,
            "max_input_chars": 60000,
            "max_deadline_seconds": 55,
            "primary_share": 0.6,
            "daily_cap_usd": 3.0,
            "cap_alert_fractions": [0.8, 1.0],
            "budget_day_tz": "Asia/Kolkata",
        }
    }
    return {
        "schema": 1,
        "app_backends": backends,
        "app_lanes": lanes,
        "app_policy_sha256": canonical_app_policy_sha256(backends, lanes),
        "release": "V85",
    }


def write_policy(path: Path, policy: dict[str, object]) -> None:
    path.write_text(json.dumps(policy), encoding="utf-8")


@pytest.mark.parametrize("entry", [[], {}, ["deepseek-direct-flash"], 1, True, None])
def test_invalid_backend_chain_entry_skips_only_its_lane(tmp_path, entry):
    policy = promoted_policy()
    policy["app_lanes"]["invalid"] = {"backend_chain": ["deepseek-direct-flash", entry]}
    policy["app_policy_sha256"] = canonical_app_policy_sha256(policy["app_backends"], policy["app_lanes"])
    path = tmp_path / "policy.json"
    write_policy(path, policy)

    loaded = validate_app_policy(policy, path)
    assert set(loaded.app_lanes) == {"chat_fast"}
    assert loaded.lane_statuses["invalid"] == {"valid": False, "reason": "invalid_lane"}


@pytest.mark.parametrize("scenario", ["disabled", "missing_parent", "not_writable", "script", "other_json", "valid"])
def test_skipped_alerts_only_write_beside_actual_writable_policy(tmp_path, monkeypatch, scenario):
    policy = promoted_policy()
    policy["app_lanes"]["invalid"] = {"backend_chain": []}
    policy["app_policy_sha256"] = canonical_app_policy_sha256(policy["app_backends"], policy["app_lanes"])
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    path = policy_dir / "custom-export.json"
    write_policy(path, policy)
    file_stat = path.stat()
    if scenario == "missing_parent":
        path = tmp_path / "missing" / "policy.json"
    elif scenario == "not_writable":
        monkeypatch.setattr(runtime_policy.os, "access", lambda *_args: False)
    elif scenario in {"script", "other_json"}:
        path = tmp_path / ("script.py" if scenario == "script" else "other.json")
        path.write_text("print('script')" if scenario == "script" else "{}", encoding="utf-8")

    loaded = validate_app_policy(policy, path, file_stat=file_stat, emit_skipped_alerts=scenario != "disabled")
    assert loaded.lane_statuses["invalid"]["reason"] == "invalid_lane"
    assert not (tmp_path / "ledger").exists()
    assert not (tmp_path / "alerts.jsonl").exists()
    assert not (tmp_path / "missing").exists()
    if scenario == "valid":
        assert (policy_dir / "ledger").is_dir()
        events = [json.loads(line) for line in (policy_dir / "alerts.jsonl").read_text().splitlines()]
        assert [event["type"] for event in events] == ["policy_lane_skipped"]
    else:
        assert not (policy_dir / "ledger").exists()
        assert not (policy_dir / "alerts.jsonl").exists()


def test_loads_harness_shaped_promoted_policy_and_reports_redacted_status(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    write_policy(path, promoted_policy())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-reported")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    loader = RuntimePolicy(path, reload_interval=0)
    policy = loader.load()
    assert policy.release == "V85"
    assert tuple(policy.app_lanes["chat_fast"]["backend_chain"]) == ("deepseek-direct-flash", "glm-openrouter-flash")
    status = app_policy_status(policy_path=path)
    assert status["package_version"] == "0.6.0"
    assert status["backends"] == [
        {"name": "deepseek-direct-flash", "key_present": True},
        {"name": "glm-openrouter-flash", "key_present": False},
    ]
    assert "not-reported" not in repr(status)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda policy: policy.update(schema=2),
        lambda policy: policy.pop("app_backends"),
        lambda policy: policy.pop("app_lanes"),
        lambda policy: policy["app_backends"]["deepseek-direct-flash"]["transport"].update(endpoint="https://unapproved.example/chat/completions", allowed_host="unapproved.example"),
        lambda policy: policy["app_backends"]["deepseek-direct-flash"]["transport"].update(endpoint="https://openrouter.ai/chat/completions"),
        lambda policy: policy["app_backends"]["deepseek-direct-flash"]["transport"].update(endpoint="http://api.deepseek.com/chat/completions"),
        lambda policy: policy["app_backends"]["deepseek-direct-flash"]["transport"].update(key_env="OPENROUTER_API_KEY"),
        lambda policy: policy["app_backends"]["deepseek-direct-flash"]["transport"].update(dialect="unapproved"),
        lambda policy: policy["app_backends"]["deepseek-direct-flash"]["transport"].update(payload_defaults={"provider": {"model": "bad"}}),
        lambda policy: policy["app_backends"]["deepseek-direct-flash"]["transport"].update(payload_defaults={"thinking": {"type": "x" * 65}}),
        lambda policy: policy["app_backends"]["deepseek-direct-flash"].update(model=""),
        lambda policy: policy["app_backends"]["deepseek-direct-flash"].update(model="model:free"),
        lambda policy: policy["app_backends"]["deepseek-direct-flash"].pop("price_card"),
        lambda policy: policy["app_lanes"]["chat_fast"].update(backend_chain=["not-declared"]),
    ],
)
def test_rejects_unsafe_policy_shapes(tmp_path, mutate):
    path = tmp_path / "policy.json"
    policy = promoted_policy()
    mutate(policy)
    # Exercise validation itself, not only the separate hash-integrity check.
    policy["app_policy_sha256"] = canonical_app_policy_sha256(
        policy.get("app_backends", {}), policy.get("app_lanes", {})
    )
    write_policy(path, policy)
    with pytest.raises(RouteUnavailable):
        RuntimePolicy(path).load()


def test_nested_payload_values_are_checked_and_policy_sha_prevents_hand_edits(tmp_path):
    path = tmp_path / "policy.json"
    policy = promoted_policy()
    policy["app_backends"]["deepseek-direct-flash"]["transport"]["payload_defaults"] = {"provider": [{"credential": "sk-secret-value"}]}
    policy["app_policy_sha256"] = canonical_app_policy_sha256(policy["app_backends"], policy["app_lanes"])
    write_policy(path, policy)
    with pytest.raises(RouteUnavailable):
        RuntimePolicy(path).load()

    policy = promoted_policy()
    policy["release"] = "edited-release"  # release is not part of the app policy digest.
    write_policy(path, policy)
    assert RuntimePolicy(path).load().release == "edited-release"
    policy["app_lanes"]["chat_fast"]["max_input_chars"] = 10
    write_policy(path, policy)
    with pytest.raises(RouteUnavailable):
        RuntimePolicy(path).load()


def test_corrupt_reload_keeps_last_good_and_alerts_are_persisted_and_deduped(tmp_path):
    path = tmp_path / "policy.json"
    write_policy(path, promoted_policy())
    seen: list[dict[str, object]] = []
    set_alert_sink(seen.append)
    try:
        loader = RuntimePolicy(path, reload_interval=0)
        good = loader.load()
        path.write_text("{not-json", encoding="utf-8")
        assert loader.load() == good
        assert loader.load() == good
    finally:
        set_alert_sink(None)
    assert len(seen) == 1
    assert seen[0]["type"] == "policy_invalid"
    assert (tmp_path / "ledger" / "alert_dedupe.json").exists()


def test_no_valid_file_fails_closed(tmp_path):
    with pytest.raises(RouteUnavailable):
        RuntimePolicy(tmp_path / "missing.json").load()


def test_policy_alert_failure_is_best_effort_and_status_is_invalid(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    write_policy(path, promoted_policy())
    loader = RuntimePolicy(path, reload_interval=0)
    good = loader.load()
    path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr("jainsons_llm_router.runtime_policy.emit_alert", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("alerts unavailable")))
    assert loader.load() == good
    status = app_policy_status(policy_path=path)
    assert status["valid"] is False


def test_reload_uses_the_open_handle_metadata(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    write_policy(path, promoted_policy())
    monkeypatch.setattr(Path, "stat", lambda self: (_ for _ in ()).throw(AssertionError("separate stat used")))
    assert RuntimePolicy(path).load().sha256


def test_missing_policy_returns_valid_false_and_alert_is_best_effort(tmp_path, monkeypatch):
    monkeypatch.setattr("jainsons_llm_router.runtime_policy.emit_alert", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("alert failed")))
    status = app_policy_status(policy_path=tmp_path / "does_not_exist.json")
    assert status["valid"] is False
    assert status["spent_today_usd"] == 0.0


def test_light_status_missing_policy_is_a_pure_read(tmp_path):
    seen = []
    set_alert_sink(seen.append)
    try:
        status = app_policy_status(policy_path=tmp_path / "does_not_exist.json", light=True)
    finally:
        set_alert_sink(None)
    assert status["valid"] is False
    assert list(tmp_path.iterdir()) == []
    assert seen == []


@pytest.mark.parametrize(
    "backend_name, defaults",
    [
        ("deepseek-direct-flash", {"provider": {"data_collection": "deny"}}),
        ("deepseek-direct-flash", {"thinking": "disabled"}),
        ("deepseek-direct-flash", {"plugins": []}),
        ("glm-openrouter-flash", {"thinking": {"type": "disabled"}}),
        ("glm-openrouter-flash", {"provider": {"models": {"bad": "nested"}}}),
        ("glm-openrouter-flash", {"provider": {"route": "fallback"}}),
        ("glm-openrouter-flash", {"provider": {"plugins": []}}),
        ("glm-openrouter-flash", {"provider": {"transforms": []}}),
        ("glm-openrouter-flash", {"models": ["z-ai/glm-5.3-flash"]}),
        ("glm-openrouter-flash", {"route": "fallback"}),
        ("glm-openrouter-flash", {"provider": {"order": {"nested": "dict"}}}),
        ("glm-openrouter-flash", {"provider": {"order": ["sk-ant-api-secret"]}}),
    ],
)
def test_payload_defaults_are_host_allow_listed(tmp_path, backend_name, defaults):
    policy = promoted_policy()
    policy["app_backends"][backend_name]["transport"]["payload_defaults"] = defaults
    policy["app_policy_sha256"] = canonical_app_policy_sha256(policy["app_backends"], policy["app_lanes"])
    path = tmp_path / "policy.json"
    write_policy(path, policy)
    with pytest.raises(RouteUnavailable):
        RuntimePolicy(path).load()


@pytest.mark.parametrize("field, value", [
    ("daily_cap_usd", 0),
    ("daily_cap_usd", -5.0),
    ("daily_cap_usd", 50.1),
    ("max_output_tokens", 0),
    ("max_output_tokens", 8001),
    ("max_input_chars", 0),
    ("max_input_chars", 500001),
    ("max_deadline_seconds", 0),
    ("max_deadline_seconds", 121),
    ("primary_share", 0),
    ("primary_share", -0.5),
    ("primary_share", 1.1),
    ("budget_day_tz", "Not/AZone"),
    ("cap_alert_fractions", []),
    ("cap_alert_fractions", [0, 1]),
    ("cap_alert_fractions", [0.5, 1.2]),
    ("backend_chain", []),
])
def test_lane_settings_are_strictly_bounded(tmp_path, field, value):
    policy = promoted_policy()
    policy["app_lanes"]["chat_fast"][field] = value
    policy["app_policy_sha256"] = canonical_app_policy_sha256(policy["app_backends"], policy["app_lanes"])
    path = tmp_path / "policy.json"
    write_policy(path, policy)
    with pytest.raises(RouteUnavailable):
        RuntimePolicy(path).load()


def test_unknown_lane_is_skipped_without_blocking_valid_chat_and_alert_is_deduped(tmp_path):
    policy = promoted_policy()
    policy["app_backends"]["future-backend"] = {"transport": {"dialect": "future-v9"}}
    policy["app_lanes"]["future_lane"] = {
        "lane_kind": "future_kind",
        "backend_chain": ["future-backend"],
    }
    policy["app_policy_sha256"] = canonical_app_policy_sha256(policy["app_backends"], policy["app_lanes"])
    path = tmp_path / "policy.json"
    write_policy(path, policy)
    seen = []
    set_alert_sink(seen.append)
    try:
        loaded = RuntimePolicy(path, reload_interval=0).load()
        RuntimePolicy(path, reload_interval=0).load()
    finally:
        set_alert_sink(None)
    assert set(loaded.app_lanes) == {"chat_fast"}
    assert "future-backend" not in loaded.app_backends
    assert loaded.lane_statuses["future_lane"]["reason"] == "unknown_lane_kind"
    assert [event["type"] for event in seen] == ["policy_lane_skipped"]
    status = app_policy_status(type="future_lane", policy_path=path)
    assert status["valid"] is False
    assert status["lane_status"] == {"valid": False, "reason": "unknown_lane_kind"}
    with pytest.raises(RouteUnavailable):
        complete_chat([{"role": "user", "content": "hello"}], type="future_lane", policy_path=path)


def test_unknown_backend_dialect_skips_only_its_referencing_lane(tmp_path):
    policy = promoted_policy()
    policy["app_backends"]["future-backend"] = {
        "kind": "app-metered",
        "model": "future/model",
        "transport": {
            "endpoint": "https://openrouter.ai/api/v1/future",
            "allowed_host": "openrouter.ai",
            "key_env": "OPENROUTER_API_KEY",
            "dialect": "future-v9",
            "payload_defaults": {},
        },
        "price_card": {"version": "future", "input": 0.1, "output": 0},
    }
    policy["app_lanes"]["future_lane"] = {
        "backend_chain": ["future-backend"],
        "max_output_tokens": 10,
        "max_input_chars": 10,
        "max_deadline_seconds": 1,
        "primary_share": 1,
        "daily_cap_usd": 1,
        "cap_alert_fractions": [1],
        "budget_day_tz": "Asia/Kolkata",
    }
    policy["app_policy_sha256"] = canonical_app_policy_sha256(policy["app_backends"], policy["app_lanes"])
    path = tmp_path / "policy.json"
    write_policy(path, policy)

    loaded = RuntimePolicy(path).load()
    assert set(loaded.app_lanes) == {"chat_fast"}
    assert loaded.lane_statuses["future_lane"] == {"valid": False, "reason": "unknown_dialect"}
    assert "future-backend" not in loaded.app_backends
