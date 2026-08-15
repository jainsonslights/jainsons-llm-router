"""Privacy-safe structured event logging."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from copy import deepcopy
from typing import Any, Mapping, Protocol


SAFE_EVENT_FIELDS = frozenset(
    {
        "event_type",
        "route",
        "provider",
        "model",
        "billing_class",
        "approval",
        "cap_scopes",
        "reservation_id",
        "latency_ms",
        "outcome",
        "reason_code",
        "attempt_number",
        "correlation_id_hash",
        "provider_request_id_hash",
        "provider_account_alias_hash",
        "service",
        "environment",
        "policy_version",
    }
)
_SECRETISH = re.compile(
    r"(?i)(?:bearer\s+[a-z0-9._~+/-]+=*|(?:sk|key|token)[-_][a-z0-9_-]{8,}|authorization\s*[:=])"
)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:/-]{1,200}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REASON_CODES = frozenset(
    {
        "zero_cost_not_proven",
        "zero_price_not_verified",
        "adapter_health_failed",
        "adapter_unavailable",
        "adapter_unhealthy",
        "credential_unavailable",
        "unsupported_model",
        "unsupported_modality",
        "invalid_timeout",
        "unknown_price_card",
        "request_build_failed",
        "provider_rejected",
        "provider_outcome_unknown",
        "invalid_provider_response",
        "sdk_adapter_not_configured",
        "paid_not_approved",
        "budget_denied",
        "idempotency_conflict",
        "adapter_contract_error",
    }
)
_HASH_FIELDS = frozenset(
    {"correlation_id_hash", "provider_request_id_hash", "provider_account_alias_hash"}
)


def stable_hash(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()


def _safe_scalar(field: str, value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if field in _HASH_FIELDS:
            return value if _HASH.fullmatch(value) else None
        if field == "reason_code":
            return value if value in _SAFE_REASON_CODES else "adapter_failure"
        cleaned = _SECRETISH.sub("[REDACTED]", value)
        cleaned = _PHONE.sub("[REDACTED]", cleaned)
        if not _SAFE_IDENTIFIER.fullmatch(cleaned):
            return "[REDACTED]"
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_safe_scalar(field, item) for item in value[:20]]
    return type(value).__name__


def sanitize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Allow-list event metadata and redact defense-in-depth patterns."""

    return {
        key: _safe_scalar(key, value)
        for key, value in event.items()
        if key in SAFE_EVENT_FIELDS
    }


class EventLogger(Protocol):
    def emit(self, event: Mapping[str, Any]) -> None: ...


class NullEventLogger:
    def emit(self, event: Mapping[str, Any]) -> None:
        return None


class StructuredEventLogger:
    """Write one sanitized JSON object per logging record."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("jainsons_llm_router")

    def emit(self, event: Mapping[str, Any]) -> None:
        self.logger.info(json.dumps(sanitize_event(event), sort_keys=True, separators=(",", ":")))


class InMemoryEventLogger:
    """Privacy-preserving event collector useful in tests and local diagnostics."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Mapping[str, Any]) -> None:
        self.events.append(deepcopy(sanitize_event(event)))
