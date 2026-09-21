"""Typed, policy-controlled decisions through the TypeSafe SystemOne API."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from .app_chat import (
    _DispatchFailure,
    _actual_cost,
    _app_spend,
    _budget_alerts,
    _perform_request,
    _price_card,
    _safe_alert,
)
from .errors import BudgetExhausted, ConfigurationError, ProviderFailure, RouteUnavailable
from .runtime_policy import get_runtime_policy


_NAME = re.compile(r"^[a-z0-9_]{1,40}$")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_GSTIN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b", re.IGNORECASE)
_LONG_NUMBER = re.compile(r"(?<!\d)\d(?:[ -]*\d){5,}(?!\d)")
_DATE_SUFFIX = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class DecisionResult:
    answers: dict[str, dict[str, Any]]
    backend: str
    model: str
    reply_model: str
    policy_sha256: str
    release: str
    elapsed_ms: int
    cost_usd: float


def _mask_pii(state: str) -> str:
    """Mask the bounded PII classes promised by the public decision API."""

    masked = _EMAIL.sub("<EMAIL>", state)
    masked = _GSTIN.sub("<GSTIN>", masked)
    return _LONG_NUMBER.sub("<NUM>", masked)


def _validate_questions(questions: Any, max_questions: int) -> dict[str, dict[str, Any]]:
    if not isinstance(questions, dict) or not questions or len(questions) > max_questions:
        raise ConfigurationError("questions must be a non-empty object within the lane limit")
    clean: dict[str, dict[str, Any]] = {}
    for name, raw in questions.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ConfigurationError("question name is invalid")
        if not isinstance(raw, dict):
            raise ConfigurationError("question must be an object")
        question_type = raw.get("type")
        instructions = raw.get("instructions")
        if question_type not in {"choice", "noul"}:
            raise ConfigurationError("question type is not supported")
        if not isinstance(instructions, str) or len(instructions) > 500:
            raise ConfigurationError("question instructions are invalid")
        allowed_fields = {"type", "instructions", "criteria"} if question_type == "choice" else {"type", "instructions"}
        if set(raw) - allowed_fields:
            raise ConfigurationError("question contains unsupported fields")
        item: dict[str, Any] = {"type": question_type, "instructions": instructions}
        if question_type == "choice":
            criteria = raw.get("criteria")
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 20:
                raise ConfigurationError("choice criteria are invalid")
            clean_criteria: dict[str, str] = {}
            for option, description in criteria.items():
                if not isinstance(option, str) or not _NAME.fullmatch(option):
                    raise ConfigurationError("choice option name is invalid")
                if not isinstance(description, str) or len(description) > 300:
                    raise ConfigurationError("choice option description is invalid")
                clean_criteria[option] = description
            item["criteria"] = clean_criteria
        clean[name] = item
    return clean


def _probability(value: Any) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ProviderFailure("decision provider returned a malformed answer")
    return float(value)


def _parse_answers(
    response: Mapping[str, Any],
    questions: Mapping[str, Mapping[str, Any]],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    raw_answers = response.get("answers")
    if not isinstance(raw_answers, dict) or set(raw_answers) != set(questions):
        raise ProviderFailure("decision provider returned malformed answers")
    answers: dict[str, dict[str, Any]] = {}
    for name, question in questions.items():
        raw = raw_answers.get(name)
        question_type = question["type"]
        if not isinstance(raw, dict) or raw.get("type") != question_type:
            raise ProviderFailure("decision provider returned a mismatched answer type")
        if question_type == "choice":
            if set(raw) != {"type", "choice", "probabilities", "confidence"}:
                raise ProviderFailure("decision provider returned a malformed choice")
            value = raw.get("choice")
            criteria = question["criteria"]
            probabilities_raw = raw.get("probabilities")
            if not isinstance(value, str) or value not in criteria or not isinstance(probabilities_raw, dict):
                raise ProviderFailure("decision provider returned a malformed choice")
            if set(probabilities_raw) - set(criteria):
                raise ProviderFailure("decision provider returned malformed choice probabilities")
            probabilities = {
                option: _probability(probabilities_raw.get(option, 0.0))
                for option in criteria
            }
            probability_sum = sum(probabilities.values())
            if probability_sum == 0 or not 0.95 <= probability_sum <= 1.05:
                raise ProviderFailure("decision provider returned malformed choice probabilities")
            probabilities = {
                option: probability / probability_sum for option, probability in probabilities.items()
            }
            confidence = _probability(raw.get("confidence"))
            answers[name] = {
                "type": "choice",
                "value": value,
                "probabilities": probabilities,
                "confidence": confidence,
                "confident": confidence >= threshold,
            }
        else:
            if set(raw) != {"type", "noul"}:
                raise ProviderFailure("decision provider returned a malformed noul")
            probability = _probability(raw.get("noul"))
            answers[name] = {
                "type": "noul",
                "value": probability >= 0.5,
                "probabilities": None,
                "p": probability,
                "confidence": max(probability, 1 - probability),
                "confident": max(probability, 1 - probability) >= threshold,
            }
    return answers


def _reply_model_matches(declared: str, reply: str) -> bool:
    if reply == declared:
        return True
    prefix = declared + "-"
    return reply.startswith(prefix) and bool(_DATE_SUFFIX.fullmatch(reply[len(prefix):]))


def _settlement_cost(
    response: Mapping[str, Any],
    *,
    estimated_input_tokens: int,
    input_rate: float,
    output_rate: float,
) -> float:
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    reported_cost = usage.get("cost")
    if (
        isinstance(reported_cost, (int, float))
        and not isinstance(reported_cost, bool)
        and math.isfinite(reported_cost)
        and reported_cost >= 0
    ):
        return float(reported_cost)
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    safe_input = input_tokens if isinstance(input_tokens, int) and not isinstance(input_tokens, bool) and input_tokens >= 0 else estimated_input_tokens
    safe_output = output_tokens if isinstance(output_tokens, int) and not isinstance(output_tokens, bool) and output_tokens >= 0 else 0
    return _actual_cost(safe_input, safe_output, input_rate, output_rate) / 1_000_000


def decide(
    state: str,
    questions: dict[str, Any],
    type: str = "decide_fast",
    *,
    profile: str = "default",
    deadline_seconds: float | None = None,
    service: str = "unknown",
    policy_path: str | os.PathLike[str] | None = None,
    spend_dir: str | os.PathLike[str] | None = None,
) -> DecisionResult:
    """Answer bounded typed questions without exposing model selection to apps.

    Noul answers contain a boolean ``value`` (``p >= 0.5``), the probability
    ``p``, and ``confidence = max(p, 1 - p)``. Choice probability totals may
    differ from one by at most 1e-3.
    """

    del service  # Reserved for privacy-safe service attribution in a later policy version.
    started = time.monotonic()
    policy = get_runtime_policy(policy_path).get()
    lane = policy.lane(type)
    if lane.get("lane_kind", "chat") != "decision":
        raise RouteUnavailable("requested app lane is not a decision lane")
    if not isinstance(state, str):
        raise ConfigurationError("state must be a string")
    if len(state) > int(lane["max_input_chars"]):
        raise ConfigurationError("decision state is too large")
    profiles = lane["min_confidence_profiles"]
    if profile not in profiles:
        raise ConfigurationError("decision confidence profile is unknown")
    threshold = float(profiles[profile])
    clean_questions = _validate_questions(questions, int(lane["max_questions"]))
    masked_state = _mask_pii(state)

    ceiling_deadline = float(lane["max_deadline_seconds"])
    if deadline_seconds is None:
        deadline = ceiling_deadline
    elif (
        not isinstance(deadline_seconds, (int, float))
        or isinstance(deadline_seconds, bool)
        or not math.isfinite(deadline_seconds)
        or deadline_seconds <= 0
    ):
        raise ConfigurationError("decision deadline is invalid")
    else:
        deadline = min(float(deadline_seconds), ceiling_deadline)
    overall_deadline = started + deadline
    correlation_id = uuid.uuid4().hex
    spend = _app_spend(policy, type, lane, spend_dir)

    for backend_name in lane["backend_chain"]:
        backend = policy.app_backends[backend_name]
        transport = backend["transport"]
        key = os.environ.get(str(transport["key_env"]), "")
        if not key or time.monotonic() >= overall_deadline:
            continue
        model = str(backend["model"])
        _price_version, input_rate, output_rate = _price_card(backend)
        estimated_input_tokens = max(1, math.ceil(len(masked_state) / 3))
        reserved_usd = max(0.000001, estimated_input_tokens * input_rate / 1_000_000)
        try:
            reservation_id = spend.reserve(reserved_usd)
        except BudgetExhausted:
            _budget_alerts(policy, type, lane, spend, backend_name, correlation_id, cap_refused=True)
            raise
        except (OSError, ValueError):
            raise RouteUnavailable("decision accounting is unavailable") from None

        payload = copy.deepcopy(dict(transport.get("payload_defaults", {})))
        payload.update({"model": model, "state": masked_state, "questions": clean_questions})
        try:
            response = _perform_request(str(transport["endpoint"]), key, payload, overall_deadline)
        except _DispatchFailure as failure:
            try:
                if failure.status == 402:
                    spend.release(reservation_id)
                    if spend.alert_once("openrouter_credit_exhausted"):
                        _safe_alert(
                            policy,
                            "provider_credit",
                            "openrouter_credit_exhausted",
                            backend_name,
                            correlation_id,
                            once_per_budget_day=True,
                            budget_day=spend._day(),
                        )
                    raise RouteUnavailable("decision provider credit is exhausted")
                if failure.pre_dispatch:
                    spend.release(reservation_id)
                else:
                    spend.settle_unknown(reservation_id)
                _budget_alerts(policy, type, lane, spend, backend_name, correlation_id)
            except RouteUnavailable:
                raise
            except Exception:
                raise RouteUnavailable("decision accounting is unavailable") from None
            continue

        try:
            reply_model = response.get("model")
            if not isinstance(reply_model, str) or not reply_model:
                raise ProviderFailure("decision provider omitted its reply model")
            answers = _parse_answers(response, clean_questions, threshold)
            cost_usd = _settlement_cost(
                response,
                estimated_input_tokens=estimated_input_tokens,
                input_rate=input_rate,
                output_rate=output_rate,
            )
            spend.settle(reservation_id, cost_usd)
            _budget_alerts(policy, type, lane, spend, backend_name, correlation_id)
        except ProviderFailure:
            try:
                spend.settle_unknown(reservation_id)
            except Exception:
                raise RouteUnavailable("decision accounting is unavailable") from None
            raise
        except (OSError, ValueError):
            raise RouteUnavailable("decision accounting is unavailable") from None

        if not _reply_model_matches(model, reply_model):
            _safe_alert(policy, "model_mismatch", "reply_model_mismatch", backend_name, correlation_id)
            raise ProviderFailure("decision provider reply model does not match declaration")
        return DecisionResult(
            answers=answers,
            backend=backend_name,
            model=model,
            reply_model=reply_model,
            policy_sha256=policy.sha256,
            release=policy.release,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            cost_usd=cost_usd,
        )

    raise RouteUnavailable("no decision backend is available")


async def adecide(*args: Any, **kwargs: Any) -> DecisionResult:
    """Run :func:`decide` in a worker thread rather than blocking an event loop."""

    return await asyncio.to_thread(decide, *args, **kwargs)


def _fallback_values(values: Any, names: list[str], questions: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(values, Mapping) or any(name not in values for name in names):
        raise ConfigurationError("decision fallback did not return every requested answer")
    for name in names:
        question = questions[name]
        if isinstance(question, Mapping) and question.get("type") == "noul" and not isinstance(values[name], bool):
            raise ConfigurationError("decision fallback noul answers must be booleans")
    return {name: values[name] for name in names}


def decide_or(
    state: str,
    questions: dict[str, Any],
    fallback: Callable[[list[str]], Mapping[str, Any]],
    *,
    type: str = "decide_fast",
    profile: str = "default",
    **kwargs: Any,
) -> dict[str, Any]:
    """Decide with one shared fallback protocol.

    ``fallback(names: list[str]) -> dict[name, value]`` is called for only the
    unsure answers. If dispatch itself fails, ``fallback(list(questions))``
    supplies all answers. Noul values are booleans, including fallback values;
    non-boolean noul fallbacks raise ConfigurationError.
    """

    try:
        result = decide(state, questions, type=type, profile=profile, **kwargs)
    except Exception as exc:
        names = list(questions) if isinstance(questions, dict) else []
        values = _fallback_values(fallback(names), names, questions)
        answers = {
            name: {"value": value, "source": "fallback", "confidence": None}
            for name, value in values.items()
        }
        return {
            "answers": answers,
            "meta": {"source": "fallback", "error": exc.__class__.__name__},
        }

    unsure = [name for name, answer in result.answers.items() if not answer["confident"]]
    fallback_answers = _fallback_values(fallback(unsure), unsure, questions) if unsure else {}
    answers: dict[str, dict[str, Any]] = {}
    for name, answer in result.answers.items():
        if name in fallback_answers:
            answers[name] = {"value": fallback_answers[name], "source": "fallback", "confidence": None}
        else:
            answers[name] = {"value": answer["value"], "source": "decide", "confidence": answer["confidence"]}
    meta = asdict(result)
    meta.pop("answers", None)
    meta["source"] = "partial_fallback" if unsure else "decide"
    return {"answers": answers, "meta": meta}


__all__ = ["DecisionResult", "adecide", "decide", "decide_or"]
