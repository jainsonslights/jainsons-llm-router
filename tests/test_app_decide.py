from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from jainsons_llm_router import ConfigurationError, ProviderFailure, RouteUnavailable, decide, decide_or, set_alert_sink
from jainsons_llm_router import app_chat, app_decide
from jainsons_llm_router.runtime_policy import canonical_app_policy_sha256


QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "Choose the category",
        "criteria": {"sales": "A sales request", "service": "A service request"},
    },
    "hot": {"type": "noul", "instructions": "Is this lead hot?"},
}


def decision_policy() -> dict[str, object]:
    backends = {
        "jev-openrouter": {
            "kind": "app-metered",
            "model": "typesafe/jev-1.13",
            "transport": {
                "endpoint": "https://openrouter.ai/api/v1/systemone",
                "allowed_host": "openrouter.ai",
                "key_env": "OPENROUTER_API_KEY",
                "dialect": "typesafe-systemone",
                "payload_defaults": {"provider": {"zdr": True, "data_collection": "deny"}},
            },
            "price_card": {
                "version": "jev-1.13-2026-09",
                "input_usd_per_million": 0.042,
                "output_usd_per_million": 0.0,
            },
        }
    }
    lanes = {
        "decide_fast": {
            "lane_kind": "decision",
            "backend_chain": ["jev-openrouter"],
            "min_confidence_profiles": {"default": 0.8, "strict": 0.9},
            "max_questions": 20,
            "max_input_chars": 20000,
            "max_deadline_seconds": 5,
            "daily_cap_usd": 1.0,
            "budget_day_tz": "Asia/Kolkata",
            "cap_alert_fractions": [0.8, 1.0],
            "quality_gate": {"version": 1},
        }
    }
    return {
        "schema": 1,
        "app_backends": backends,
        "app_lanes": lanes,
        "app_policy_sha256": canonical_app_policy_sha256(backends, lanes),
        "release": "V87",
    }


def policy_file(tmp_path: Path, mutate=None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    policy = decision_policy()
    if mutate:
        mutate(policy)
    policy["app_policy_sha256"] = canonical_app_policy_sha256(policy["app_backends"], policy["app_lanes"])
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    return path


def good_response(*, confidence=0.91, p=0.1, model="typesafe/jev-1.13-20260917", cost=0.00003):
    return {
        "model": model,
        "answers": {
            "category": {
                "type": "choice",
                "choice": "sales",
                "probabilities": {"sales": 0.95, "service": 0.05},
                "confidence": confidence,
            },
            "hot": {"type": "noul", "noul": p},
        },
        "usage": {"input_tokens": 100, "output_tokens": 10, "cost": cost},
    }


def test_decide_masks_pii_builds_bound_payload_and_parses_confidence(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-not-reported")
    captured = {}

    def fake_request(endpoint, key, payload, deadline):
        captured.update(endpoint=endpoint, key=key, payload=payload, deadline=deadline)
        return good_response()

    monkeypatch.setattr(app_decide, "_perform_request", fake_request)
    result = decide(
        "mail me at person@example.com GST 27ABCDE1234F1Z5 phone 98765-43210",
        QUESTIONS,
        profile="strict",
        policy_path=path,
    )
    assert captured["endpoint"] == "https://openrouter.ai/api/v1/systemone"
    assert captured["payload"]["state"] == "mail me at <EMAIL> GST <GSTIN> phone <NUM>"
    assert captured["payload"]["provider"] == {"zdr": True, "data_collection": "deny"}
    assert result.answers["category"]["confident"] is True
    assert result.answers["hot"] == {
        "type": "noul", "value": False, "probabilities": None, "p": 0.1, "confidence": 0.9, "confident": True,
    }
    assert result.reply_model.endswith("-20260917")
    assert result.cost_usd == 0.00003
    spend_files = list((tmp_path / "spend").glob("decide_fast-*.json"))
    assert len(spend_files) == 1
    assert json.loads(spend_files[0].read_text())["spent_usd"] == 0.00003


@pytest.mark.parametrize(
    "mutate",
    [
        lambda response: response["answers"]["category"].update(type="noul"),
        lambda response: response["answers"]["category"].update(choice="other"),
        lambda response: response["answers"]["category"].update(extra="unexpected"),
        lambda response: response["answers"]["hot"].update(noul=1.2),
        lambda response: response["answers"]["hot"].update(confidence=0.9),
        lambda response: response["answers"].pop("hot"),
    ],
)
def test_malformed_provider_answers_raise_provider_failure(tmp_path, monkeypatch, mutate):
    path = policy_file(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    response = good_response()
    mutate(response)
    monkeypatch.setattr(app_decide, "_perform_request", lambda *_args: response)
    with pytest.raises(ProviderFailure):
        decide("safe state", QUESTIONS, policy_path=path)


def _choice_parse_response(probabilities):
    return {
        "answers": {
            "category": {
                "type": "choice",
                "choice": "sales",
                "probabilities": probabilities,
                "confidence": 0.9,
            }
        }
    }


def test_choice_probabilities_accept_missing_zero_option():
    answers = app_decide._parse_answers(
        _choice_parse_response({"sales": 1.0}),
        {"category": QUESTIONS["category"]},
        0.8,
    )
    assert answers["category"]["probabilities"] == {"sales": 1.0, "service": 0.0}


def test_choice_probabilities_reject_extra_option():
    with pytest.raises(ProviderFailure, match="malformed choice probabilities"):
        app_decide._parse_answers(
            _choice_parse_response({"sales": 0.95, "service": 0.05, "other": 0.0}),
            {"category": QUESTIONS["category"]},
            0.8,
        )


def test_question_and_profile_validation_happens_before_network(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(app_decide, "_perform_request", lambda *_args: pytest.fail("network called"))
    with pytest.raises(ConfigurationError):
        decide("safe", QUESTIONS, profile="missing", policy_path=path)
    bad = {"bad-name": {"type": "noul", "instructions": "yes?"}}
    with pytest.raises(ConfigurationError):
        decide("safe", bad, policy_path=path)
    score = {"score": {"type": "score", "instructions": "rate", "criteria": ["a", "b"]}}
    with pytest.raises(ConfigurationError):
        decide("safe", score, policy_path=path)


def test_empty_state_and_instructions_are_valid_bounded_inputs(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(app_decide, "_perform_request", lambda *_args: good_response())
    questions = {
        "category": {
            "type": "choice",
            "instructions": "",
            "criteria": {"sales": "", "service": ""},
        },
        "hot": {"type": "noul", "instructions": ""},
    }
    assert decide("", questions, policy_path=path).answers["category"]["value"] == "sales"


def test_reply_model_mismatch_is_alerted_and_falls_back_to_the_caller(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(app_decide, "_perform_request", lambda *_args: good_response(model="other/model"))
    seen = []
    set_alert_sink(seen.append)
    try:
        with pytest.raises(ProviderFailure, match="reply model"):
            decide("safe", QUESTIONS, policy_path=path)
    finally:
        set_alert_sink(None)
    assert [event["reason"] for event in seen] == ["reply_model_mismatch"]


def test_decide_or_handles_total_failure_and_only_unsure_subset(monkeypatch):
    calls = []

    def all_fallback(names):
        calls.append(names)
        return {"category": "service", "hot": True}

    monkeypatch.setattr(app_decide, "decide", lambda *_args, **_kwargs: (_ for _ in ()).throw(RouteUnavailable()))
    failed = decide_or("state", QUESTIONS, all_fallback)
    assert calls == [["category", "hot"]]
    assert failed["answers"]["category"] == {"value": "service", "source": "fallback", "confidence": None}
    assert failed["answers"]["hot"]["value"] is True

    calls.clear()
    result = app_decide.DecisionResult(
        answers={
            "category": {"type": "choice", "value": "sales", "probabilities": {}, "confidence": 0.95, "confident": True},
            "hot": {"type": "noul", "value": True, "probabilities": None, "p": 0.55, "confidence": 0.55, "confident": False},
        },
        backend="jev", model="m", reply_model="m", policy_sha256="a" * 64,
        release="V87", elapsed_ms=1, cost_usd=0.1,
    )
    monkeypatch.setattr(app_decide, "decide", lambda *_args, **_kwargs: result)

    def subset_fallback(names):
        calls.append(names)
        return {"hot": False}

    partial = decide_or("state", QUESTIONS, subset_fallback)
    assert calls == [["hot"]]
    assert partial["answers"]["category"]["source"] == "decide"
    assert partial["answers"]["hot"] == {"value": False, "source": "fallback", "confidence": None}


@pytest.mark.parametrize("p", [0.0, 0.1, 0.2, 0.49, 0.5, 0.51, 0.8, 0.9, 1.0])
def test_noul_boolean_probability_and_confidence(tmp_path, monkeypatch, p):
    path = policy_file(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(app_decide, "_perform_request", lambda *_args: good_response(p=p))
    result = decide("safe", QUESTIONS, policy_path=path)
    answer = result.answers["hot"]
    assert answer["value"] is (p >= 0.5)
    assert answer["p"] == p
    assert answer["confidence"] == max(p, 1 - p)
    assert answer["confident"] is (max(p, 1 - p) >= 0.8)

    calls = []

    def fallback(names):
        calls.append(names)
        return {"hot": False}

    monkeypatch.setattr(app_decide, "decide", lambda *_args, **_kwargs: result)
    wrapped = decide_or("safe", QUESTIONS, fallback)
    if answer["confident"]:
        assert calls == []
        assert wrapped["answers"]["hot"] == {
            "value": p >= 0.5, "source": "decide", "confidence": max(p, 1 - p),
        }
        assert wrapped["answers"]["hot"]["value"] is (p >= 0.5)
    else:
        assert calls == [["hot"]]
        assert wrapped["answers"]["hot"]["value"] is False


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("value", [0.1, 0, 1, "false", None])
def test_decide_or_rejects_non_boolean_noul_fallbacks(tmp_path, monkeypatch, failed, value):
    path = policy_file(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    if failed:
        monkeypatch.setattr(app_decide, "decide", lambda *_args, **_kwargs: (_ for _ in ()).throw(RouteUnavailable()))
    else:
        monkeypatch.setattr(app_decide, "_perform_request", lambda *_args: good_response(p=0.5))
    with pytest.raises(ConfigurationError, match="noul answers must be booleans"):
        decide_or("safe", QUESTIONS, lambda names: {"category": "sales", "hot": value}, policy_path=path)


@pytest.mark.parametrize("total", [0.98, 1.0, 1.02])
def test_choice_probability_sum_tolerates_rounding_and_normalises(tmp_path, monkeypatch, total):
    path = policy_file(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    response = good_response()
    probabilities = {"sales": 0.5, "service": total - 0.5}
    response["answers"]["category"]["probabilities"] = probabilities
    monkeypatch.setattr(app_decide, "_perform_request", lambda *_args: response)
    parsed = decide("safe", QUESTIONS, policy_path=path).answers["category"]["probabilities"]
    assert sum(parsed.values()) == pytest.approx(1.0)
    assert parsed["sales"] == pytest.approx(probabilities["sales"] / total)
    assert parsed["service"] == pytest.approx(probabilities["service"] / total)


@pytest.mark.parametrize("total", [0.9, 1.1])
def test_choice_probability_sum_outside_tolerance_is_rejected(tmp_path, monkeypatch, total):
    path = policy_file(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    response = good_response()
    response["answers"]["category"]["probabilities"] = {"sales": 0.5, "service": total - 0.5}
    monkeypatch.setattr(app_decide, "_perform_request", lambda *_args: response)
    with pytest.raises(ProviderFailure, match="choice probabilities"):
        decide("safe", QUESTIONS, policy_path=path)


def test_http_402_releases_reservation_and_alerts_once_per_day(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(
        app_decide,
        "_perform_request",
        lambda *_args: (_ for _ in ()).throw(app_chat._DispatchFailure("http_error", ambiguous=True, status=402)),
    )
    seen = []
    set_alert_sink(seen.append)
    try:
        for _ in range(2):
            with pytest.raises(RouteUnavailable, match="credit"):
                decide("safe", QUESTIONS, policy_path=path)
    finally:
        set_alert_sink(None)
    assert [item["reason"] for item in seen] == ["openrouter_credit_exhausted"]
    state = json.loads(next((tmp_path / "spend").glob("decide_fast-*.json")).read_text())
    assert state["spent_usd"] == 0
    assert state["reservations"] == {}


def test_decision_lane_and_typesafe_transport_are_strict(tmp_path):
    def wrong_endpoint(policy):
        policy["app_backends"]["jev-openrouter"]["transport"]["endpoint"] = "https://openrouter.ai/api/v1/chat/completions"

    path = policy_file(tmp_path, wrong_endpoint)
    with pytest.raises(RouteUnavailable):
        decide("safe", QUESTIONS, policy_path=path)

    def wrong_zdr(policy):
        policy["app_backends"]["jev-openrouter"]["transport"]["payload_defaults"]["provider"]["zdr"] = "true"

    path = policy_file(tmp_path / "second", wrong_zdr)
    with pytest.raises(RouteUnavailable):
        decide("safe", QUESTIONS, policy_path=path)


def test_real_read_loop_enforces_hard_deadline_on_trickle(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")

    class Trickle:
        status = 200

        def __init__(self):
            self.closed = False

        def settimeout(self, _value):
            pass

        def read1(self, _size):
            time.sleep(0.03)
            return b" "

        def close(self):
            self.closed = True

    class Opener:
        def open(self, *_args, **_kwargs):
            return Trickle()

    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: Opener())
    started = time.monotonic()
    with pytest.raises(RouteUnavailable):
        decide("safe", QUESTIONS, deadline_seconds=0.08, policy_path=path)
    assert time.monotonic() - started < 0.5
