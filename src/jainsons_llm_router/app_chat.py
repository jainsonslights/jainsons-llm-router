"""The paid, policy-controlled application chat lane."""

from __future__ import annotations

import asyncio
import copy
import errno
import http.client
import json
import math
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ._version import __version__
from .app_alerts import emit_alert, set_alert_sink
from .app_spend import AppSpend
from .errors import BudgetExhausted, ConfigurationError, RouteUnavailable
from .runtime_policy import get_runtime_policy


_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_MIN_FALLBACK_SECONDS = 2.0


@dataclass(frozen=True)
class ChatResult:
    text: str
    backend: str
    model: str
    reply_model: str
    policy_sha256: str
    release: str
    attempts: int
    elapsed_ms: int
    cost_usd: float


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


class _DispatchFailure(Exception):
    def __init__(
        self,
        reason: str,
        *,
        ambiguous: bool = False,
        pre_dispatch: bool = False,
        status: int | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.ambiguous = ambiguous
        self.pre_dispatch = pre_dispatch
        self.status = status


def _opener() -> Any:
    """Create an opener which cannot follow a provider redirect."""

    return urllib.request.build_opener(_NoRedirectHandler())


def _set_socket_timeout(response: Any, timeout: float) -> None:
    for candidate in (
        response,
        getattr(response, "_sock", None),
        getattr(response, "sock", None),
        getattr(response, "raw", None),
        getattr(getattr(response, "raw", None), "_sock", None),
        getattr(getattr(response, "raw", None), "sock", None),
        getattr(response, "fp", None),
        getattr(getattr(response, "fp", None), "raw", None),
        getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None),
        getattr(getattr(getattr(response, "fp", None), "raw", None), "sock", None),
        getattr(getattr(response, "fp", None), "_sock", None),
        getattr(getattr(response, "fp", None), "sock", None),
    ):
        setter = getattr(candidate, "settimeout", None)
        if callable(setter):
            try:
                setter(max(0.001, timeout))
            except OSError:
                pass


def _read_response(response: Any, deadline: float) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _DispatchFailure("deadline", ambiguous=True)
            # The socket timeout is the whole remaining attempt budget.  A short
            # fixed slice turns a harmless provider pause into a false failure.
            _set_socket_timeout(response, remaining)
            reader = getattr(response, "read1", None) or getattr(response, "read")
            try:
                chunk = reader(min(_READ_CHUNK_BYTES, _MAX_RESPONSE_BYTES - total + 1))
            except (TimeoutError, socket.timeout):
                raise _DispatchFailure("read_timeout", ambiguous=True) from None
            except (OSError, ValueError, http.client.HTTPException):
                raise _DispatchFailure("read_error", ambiguous=True) from None
            if time.monotonic() >= deadline:
                raise _DispatchFailure("deadline", ambiguous=True)
            if not isinstance(chunk, (bytes, bytearray)):
                raise _DispatchFailure("invalid_response", ambiguous=True)
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > _MAX_RESPONSE_BYTES:
                raise _DispatchFailure("response_too_large", ambiguous=True)
            chunks.append(bytes(chunk))
            if time.monotonic() >= deadline:
                raise _DispatchFailure("deadline", ambiguous=True)
    finally:
        try:
            response.close()
        except Exception:
            pass


def _response_status(response: Any) -> int:
    value = getattr(response, "status", None)
    if value is None:
        getter = getattr(response, "getcode", None)
        value = getter() if callable(getter) else None
    return int(value or 0)


def _perform_request(endpoint: str, key: str, payload: Mapping[str, Any], deadline: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _DispatchFailure("deadline", ambiguous=True)
    try:
        response = _opener().open(request, timeout=remaining)
    except urllib.error.HTTPError as exc:
        status = int(getattr(exc, "code", 0) or 0)
        try:
            exc.close()
        except Exception:
            pass
        raise _DispatchFailure("http_error", ambiguous=True, status=status) from None
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        refused = isinstance(reason, ConnectionRefusedError) or getattr(reason, "errno", None) == errno.ECONNREFUSED
        definite_pre_send = refused or isinstance(reason, (socket.gaierror, ssl.SSLError))
        if definite_pre_send:
            raise _DispatchFailure(
                "connect_refused" if refused else "connect_error", pre_dispatch=True
            ) from None
        raise _DispatchFailure("open_error", ambiguous=True) from None
    except (TimeoutError, socket.timeout):
        raise _DispatchFailure("open_timeout", ambiguous=True) from None
    except http.client.HTTPException:
        raise _DispatchFailure("open_error", ambiguous=True) from None
    except OSError:
        raise _DispatchFailure("open_error", ambiguous=True) from None

    status = _response_status(response)
    if status != 200:
        try:
            response.close()
        except Exception:
            pass
        raise _DispatchFailure("http_error", ambiguous=True, status=status)
    raw = _read_response(response, deadline)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _DispatchFailure("invalid_json", ambiguous=True) from None
    if not isinstance(parsed, dict):
        raise _DispatchFailure("invalid_json", ambiguous=True)
    return parsed


def _price_card(backend: Mapping[str, Any]) -> tuple[str, float, float]:
    card = backend["price_card"]
    return (
        str(card["version"]),
        float(card.get("input_usd_per_million", card.get("input"))),
        float(card.get("output_usd_per_million", card.get("output"))),
    )


def _app_spend(policy: Any, lane_name: str, lane: Mapping[str, Any], spend_dir: str | os.PathLike[str] | None) -> AppSpend:
    return AppSpend(
        spend_dir or policy.path.parent / "spend",
        lane_name,
        float(lane["daily_cap_usd"]),
        str(lane["budget_day_tz"]),
    )


def _safe_alert(policy: Any, event_type: str, reason: str, backend: str, correlation_id: str, **kwargs: Any) -> None:
    event = {
        "type": event_type,
        "reason": reason,
        "backend": backend,
        "policy_sha256": policy.sha256,
        "release": policy.release,
        "correlation_id": correlation_id,
    }
    try:
        emit_alert(policy.path.parent, event, **kwargs)
    except Exception:
        pass


def _budget_alerts(policy: Any, lane_name: str, lane: Mapping[str, Any], spend: AppSpend, backend: str, correlation_id: str, *, cap_refused: bool = False) -> None:
    try:
        cap = float(lane["daily_cap_usd"])
        spent_fraction = spend.spent_today() / cap if cap else 1.0
        for fraction in lane["cap_alert_fractions"]:
            alert_name = f"budget_{int(float(fraction) * 100)}"
            if (spent_fraction >= float(fraction) or (cap_refused and float(fraction) == 1.0)) and spend.alert_once(alert_name):
                _safe_alert(
                    policy,
                    "budget_threshold",
                    alert_name,
                    backend,
                    correlation_id,
                    once_per_budget_day=True,
                    budget_day=spend._day(),
                )
    except Exception:
        pass


def _model_matches(declared: str, reply: str) -> bool:
    if reply == declared:
        return True
    return reply.endswith("/" + declared) or reply.startswith(declared + ":") or reply.startswith(declared + "-")


def _safe_identifier(value: str) -> str:
    result = "".join(char if char.isalnum() or char in "._:/-" else "-" for char in value)
    return result[:200] or "unknown"


def _actual_cost(input_tokens: int, output_tokens: int, input_rate: float, output_rate: float) -> int:
    # Rates are USD per million tokens, so token * rate is already micro-USD.
    return max(1, math.ceil(input_tokens * input_rate + output_tokens * output_rate))


def _validate_messages(messages: Sequence[Mapping[str, Any]], max_input_chars: int) -> tuple[list[dict[str, str]], int]:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        raise ConfigurationError("messages must be a non-empty sequence")
    allowed_roles = {"system", "user", "assistant"}
    clean: list[dict[str, str]] = []
    total = 0
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") not in allowed_roles:
            raise ConfigurationError("message role is invalid")
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise ConfigurationError("message content is invalid")
        total += len(content)
        if total > max_input_chars:
            raise ConfigurationError("message input is too large")
        clean.append({"role": str(message["role"]), "content": content})
    return clean, total


def complete_chat(
    messages: Sequence[Mapping[str, Any]],
    type: str = "chat_fast",
    *,
    max_output_tokens: int | None = None,
    deadline_seconds: float | None = None,
    service: str = "unknown",
    policy_path: str | os.PathLike[str] | None = None,
    spend_dir: str | os.PathLike[str] | None = None,
) -> ChatResult:
    started = time.monotonic()
    runtime = get_runtime_policy(policy_path)
    policy = runtime.get()
    lane = policy.lane(type)
    if lane.get("lane_kind", "chat") != "chat":
        raise RouteUnavailable("requested app lane is not a chat lane")
    clean_messages, input_chars = _validate_messages(messages, int(lane["max_input_chars"]))
    ceiling_tokens = int(lane["max_output_tokens"])
    output_tokens = ceiling_tokens if max_output_tokens is None else max(1, min(int(max_output_tokens), ceiling_tokens))
    ceiling_deadline = float(lane["max_deadline_seconds"])
    deadline = ceiling_deadline if deadline_seconds is None else max(0.001, min(float(deadline_seconds), ceiling_deadline))
    overall_deadline = started + deadline
    correlation_id = uuid.uuid4().hex
    spend = _app_spend(policy, type, lane, spend_dir)
    attempts = 0
    usable_indices = [
        index
        for index, backend_name in enumerate(lane["backend_chain"])
        if os.environ.get(str(policy.app_backends[backend_name]["transport"]["key_env"]), "")
    ]
    first_usable_index = usable_indices[0] if usable_indices else None

    for index, backend_name in enumerate(lane["backend_chain"]):
        backend = policy.app_backends[backend_name]
        transport = backend["transport"]
        key = os.environ.get(str(transport["key_env"]), "")
        if not key:
            continue
        now = time.monotonic()
        has_later_usable_backend = any(candidate > index for candidate in usable_indices)
        if index == first_usable_index:
            attempt_deadline = (
                min(overall_deadline, started + deadline * float(lane["primary_share"]))
                if has_later_usable_backend
                else overall_deadline
            )
        else:
            remaining = overall_deadline - now
            if remaining < _MIN_FALLBACK_SECONDS:
                continue
            attempt_deadline = overall_deadline
        if attempt_deadline - now <= 0:
            continue

        model = str(backend["model"])
        _price_version, input_rate, output_rate = _price_card(backend)
        request_tokens = max(output_tokens, int(backend.get("min_request_max_tokens", 1)))
        estimated_input_tokens = max(1, math.ceil(input_chars / 3))
        reserved_usd = max(0.000001, (estimated_input_tokens * input_rate + request_tokens * output_rate) / 1_000_000)
        try:
            reservation_id = spend.reserve(reserved_usd)
        except BudgetExhausted:
            _budget_alerts(policy, type, lane, spend, backend_name, correlation_id, cap_refused=True)
            raise
        except (OSError, ValueError):
            raise RouteUnavailable("application chat route is unavailable") from None

        attempts += 1
        payload = copy.deepcopy(dict(transport.get("payload_defaults", {})))
        payload.update({"model": model, "messages": clean_messages, "max_tokens": request_tokens})
        try:
            response = _perform_request(str(transport["endpoint"]), key, payload, attempt_deadline)
            choices = response.get("choices")
            message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
            text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(text, str) or not text:
                raise _DispatchFailure("empty_response", ambiguous=True)
            reply_model = response.get("model", model)
            if not isinstance(reply_model, str):
                reply_model = model
            if not _model_matches(model, reply_model):
                _safe_alert(policy, "model_mismatch", "reply_model_mismatch", backend_name, correlation_id)
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            in_tokens = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else estimated_input_tokens
            out_tokens = usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else max(1, math.ceil(len(text) / 4))
            actual_micro = _actual_cost(max(0, in_tokens), max(0, out_tokens), input_rate, output_rate)
            spend.settle(reservation_id, actual_micro / 1_000_000)
            _budget_alerts(policy, type, lane, spend, backend_name, correlation_id)
            return ChatResult(
                text=text,
                backend=backend_name,
                model=model,
                reply_model=reply_model,
                policy_sha256=policy.sha256,
                release=policy.release,
                attempts=attempts,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                cost_usd=actual_micro / 1_000_000,
            )
        except _DispatchFailure as failure:
            try:
                if failure.ambiguous:
                    spend.settle_unknown(reservation_id)
                else:
                    spend.release(reservation_id)
                _budget_alerts(policy, type, lane, spend, backend_name, correlation_id)
            except Exception:
                raise RouteUnavailable("application chat accounting is unavailable") from None
        except (BudgetExhausted, ConfigurationError):
            raise RouteUnavailable("application chat accounting is unavailable") from None

    raise RouteUnavailable("no application chat backend is available") from None


async def acomplete_chat(*args: Any, **kwargs: Any) -> ChatResult:
    """Run the synchronous dispatcher off the event loop."""

    return await asyncio.to_thread(complete_chat, *args, **kwargs)


def app_policy_status(type: str = "chat_fast", *, policy_path: str | os.PathLike[str] | None = None, light: bool = False) -> dict[str, Any]:
    """Return redacted policy readiness, key presence, and shared daily spend.

    Returns a dictionary containing:
        package_version (str): The installed version of jainsons-llm-router.
        policy_path (str): The filesystem path to the promoted policy JSON.
        valid (bool): True if the promoted policy loaded successfully and is valid.
        sha (str): SHA-256 hash of the canonical app policy, or empty string if invalid.
        release (str): Harness release label, or empty string if invalid.
        loaded_at (str | None): ISO UTC timestamp when the policy was loaded, or None.
        backends (list[dict[str, Any]]): List of backends in the lane's chain with
            'name' (str) and 'key_present' (bool).
        spent_today_usd (float): Total USD spent today in the lane's budget timezone.
        cap_usd (float | None): Daily spend cap in USD for this lane.
    """

    runtime = get_runtime_policy(policy_path)
    try:
        policy = runtime.read_only() if light else runtime.get()
    except Exception:
        return {
            "package_version": __version__, "policy_path": str(runtime.path), "valid": False,
            "sha": "", "release": "", "loaded_at": None, "backends": [],
            "spent_today_usd": 0.0, "cap_usd": None, "lanes": {}, "lane_status": None,
        }
    lanes = {name: dict(status) for name, status in policy.lane_statuses.items()}
    selected_status = lanes.get(type, {"valid": False, "reason": "unknown_lane"})
    try:
        lane = policy.lane(type)
    except Exception:
        return {
            "package_version": __version__, "policy_path": str(policy.path), "valid": False,
            "sha": policy.sha256, "release": policy.release, "loaded_at": policy.loaded_at,
            "backends": [], "spent_today_usd": 0.0, "cap_usd": None,
            "lanes": lanes, "lane_status": selected_status,
        }
    if not light and runtime.last_load_failed:
        return {
            "package_version": __version__, "policy_path": str(runtime.path), "valid": False,
            "sha": "", "release": "", "loaded_at": None, "backends": [],
            "spent_today_usd": 0.0, "cap_usd": None,
            "lanes": lanes, "lane_status": selected_status,
        }
    try:
        backends = [
            {"name": name, "key_present": bool(os.environ.get(policy.app_backends[name]["transport"]["key_env"]))}
            for name in lane["backend_chain"]
        ]
        cap = lane["daily_cap_usd"]
        spent = 0.0 if light else _app_spend(policy, type, lane, None).spent_today(read_only=True)
    except Exception:
        return {
            "package_version": __version__, "policy_path": str(runtime.path), "valid": False,
            "sha": "", "release": "", "loaded_at": None, "backends": [],
            "spent_today_usd": 0.0, "cap_usd": None,
            "lanes": lanes, "lane_status": selected_status,
        }
    return {
        "package_version": __version__, "policy_path": str(policy.path), "valid": True,
        "sha": policy.sha256, "release": policy.release, "loaded_at": policy.loaded_at,
        "backends": backends, "spent_today_usd": spent,
        "cap_usd": float(cap) if isinstance(cap, (int, float)) and not isinstance(cap, bool) else None,
        "lanes": lanes, "lane_status": selected_status,
    }


__all__ = [
    "BudgetExhausted", "ChatResult", "acomplete_chat", "app_policy_status",
    "complete_chat", "set_alert_sink",
]
