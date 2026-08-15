"""Public request, policy, and result models.

The objects are immutable so trusted deployment configuration cannot be
silently changed by a request handler after the router is created.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import ConfigurationError, InvalidRequest


_SAFE_IDENTIFIER_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:/-"
)


def validate_identifier(value: str, label: str) -> str:
    if not value or len(value) > 200 or any(ch not in _SAFE_IDENTIFIER_CHARS for ch in value):
        raise ConfigurationError(f"{label} must be a non-empty privacy-safe identifier")
    return value


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


class BillingClass(str, Enum):
    FREE = "free"
    PAID = "paid"


class FailurePhase(str, Enum):
    PRE_DISPATCH = "pre_dispatch"
    DEFINITIVE = "definitive"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LLMRequest:
    """A provider-neutral generation request.

    Set exactly one of ``input`` and ``messages``. An idempotency key is only
    required if execution actually reaches a paid candidate.
    """

    input: str | None = None
    messages: Sequence[Mapping[str, Any]] | None = None
    task_kind: str = "text_generation"
    modality: str = "text"
    response_format: str | None = None
    generation: Mapping[str, Any] = field(default_factory=dict)
    max_output_tokens: int = 512
    idempotency_key: str | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if (self.input is None) == (self.messages is None):
            raise InvalidRequest("set exactly one of input or messages")
        if self.messages is not None and not self.messages:
            raise InvalidRequest("messages cannot be empty")
        if not isinstance(self.max_output_tokens, int) or isinstance(self.max_output_tokens, bool):
            raise InvalidRequest("max_output_tokens must be an integer")
        if self.max_output_tokens <= 0:
            raise InvalidRequest("max_output_tokens must be positive")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise InvalidRequest("timeout_seconds must be positive")
        validate_identifier(self.task_kind, "task_kind")
        validate_identifier(self.modality, "modality")
        if self.response_format is not None:
            validate_identifier(self.response_format, "response_format")
        object.__setattr__(self, "generation", _immutable_mapping(self.generation))
        if self.messages is not None:
            object.__setattr__(self, "messages", tuple(MappingProxyType(dict(item)) for item in self.messages))

    @property
    def input_size(self) -> int:
        if self.input is not None:
            return len(self.input)
        assert self.messages is not None
        return sum(len(str(item.get("content", ""))) for item in self.messages)


@dataclass(frozen=True)
class CallerContext:
    service: str
    environment: str
    route_purpose: str
    deployment_version: str
    correlation_id: str
    caller_id: str = ""

    def __post_init__(self) -> None:
        validate_identifier(self.service, "service")
        validate_identifier(self.environment, "environment")
        validate_identifier(self.route_purpose, "route_purpose")
        validate_identifier(self.deployment_version, "deployment_version")
        if not self.correlation_id or len(self.correlation_id) > 1024:
            raise InvalidRequest("correlation_id is required and bounded")
        if self.caller_id:
            validate_identifier(self.caller_id, "caller_id")

    @property
    def identity(self) -> str:
        return self.caller_id or self.service


@dataclass(frozen=True)
class UseRoute:
    name: str = "default"

    def __post_init__(self) -> None:
        validate_identifier(self.name, "route name")


@dataclass(frozen=True)
class ExactModel:
    provider: str
    model: str

    def __post_init__(self) -> None:
        validate_identifier(self.provider, "provider")
        validate_identifier(self.model, "model")


RouteSelection = UseRoute | ExactModel


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token usage cannot be negative")


@dataclass(frozen=True)
class AttemptSummary:
    provider: str
    model: str
    billing_class: BillingClass
    outcome: str
    reason_code: str | None = None
    elapsed_ms: int = 0


@dataclass(frozen=True)
class LLMResult:
    output: Any
    provider: str
    model: str
    usage: Usage | None
    provider_request_id: str | None
    billing_class: BillingClass
    reservation_id: str | None
    elapsed_ms: int
    attempts: tuple[AttemptSummary, ...]

    @property
    def paid(self) -> bool:
        return self.billing_class is BillingClass.PAID


@dataclass(frozen=True)
class BudgetCap:
    calls: int
    micro_usd: int

    def __post_init__(self) -> None:
        for label, value in (("calls", self.calls), ("micro_usd", self.micro_usd)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ConfigurationError(f"{label} cap must be a positive integer")


@dataclass(frozen=True)
class BudgetRemaining:
    scope: str
    budget_day: str
    calls_remaining: int
    micro_usd_remaining: int
    calls_limit: int
    micro_usd_limit: int


@dataclass(frozen=True)
class Candidate:
    provider: str
    model: str
    adapter: str
    billing_class: BillingClass
    provider_account_alias: str
    aggregate_scope: str = ""
    provider_scope: str = ""
    route_scope: str = ""
    model_scope: str | None = None
    price_card_version: str | None = None
    zero_marginal_cost: bool = False
    retry_attempts: int = 1
    timeout_seconds: float = 30.0
    model_family: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.billing_class, BillingClass):
            raise ConfigurationError("candidate billing_class must be BillingClass.FREE or BillingClass.PAID")
        for label, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("adapter", self.adapter),
            ("provider_account_alias", self.provider_account_alias),
        ):
            validate_identifier(value, label)
        if not isinstance(self.retry_attempts, int) or isinstance(self.retry_attempts, bool) or self.retry_attempts <= 0:
            raise ConfigurationError("retry_attempts must be positive")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ConfigurationError("candidate timeout_seconds must be positive")
        if self.billing_class is BillingClass.PAID:
            required = {
                "aggregate_scope": self.aggregate_scope,
                "provider_scope": self.provider_scope,
                "route_scope": self.route_scope,
                "price_card_version": self.price_card_version or "",
            }
            for label, value in required.items():
                validate_identifier(value, label)
            if self.model_scope:
                validate_identifier(self.model_scope, "model_scope")

    @property
    def scopes(self) -> tuple[str, ...]:
        values = [self.aggregate_scope, self.provider_scope, self.route_scope]
        if self.model_scope:
            values.append(self.model_scope)
        return tuple(values)


def _forbid_glm_automatic(candidate: Candidate) -> None:
    """Make GLM impossible to place in an automatic candidate collection."""

    provider = candidate.provider.casefold()
    model = candidate.model.casefold()
    if provider == "glm" or provider.startswith("glm-") or "glm" in model:
        raise ConfigurationError("GLM cannot be an automatic route candidate")


@dataclass(frozen=True)
class RoutePolicy:
    name: str
    free_candidates: tuple[Candidate, ...] = ()
    paid_candidates: tuple[Candidate, ...] = ()
    exact_candidates: tuple[Candidate, ...] = ()
    allowed_callers: frozenset[str] = frozenset()
    allowed_environments: frozenset[str] = frozenset()
    accepted_task_kinds: frozenset[str] = frozenset({"text_generation"})
    accepted_modalities: frozenset[str] = frozenset({"text"})
    max_input_size: int = 100_000
    max_output_tokens: int = 4096
    paid_allowed: bool = False
    substitutions_allowed: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.name, "route name")
        if self.max_input_size <= 0 or self.max_output_tokens <= 0:
            raise ConfigurationError("route size limits must be positive")
        for candidate in self.free_candidates:
            _forbid_glm_automatic(candidate)
            if candidate.billing_class is not BillingClass.FREE:
                raise ConfigurationError("free_candidates must be classified free")
        for candidate in self.paid_candidates:
            _forbid_glm_automatic(candidate)
            if candidate.billing_class is not BillingClass.PAID:
                raise ConfigurationError("paid_candidates must be classified paid")
        for candidate in self.exact_candidates:
            if candidate.billing_class not in {BillingClass.FREE, BillingClass.PAID}:
                raise ConfigurationError("exact_candidates must have a known billing classification")
        for value in self.allowed_callers:
            validate_identifier(value, "allowed caller")
        for value in self.allowed_environments:
            validate_identifier(value, "allowed environment")

    def find_exact(self, provider: str, model: str) -> Candidate | None:
        candidates = self.exact_candidates + self.free_candidates + self.paid_candidates
        return next((c for c in candidates if c.provider == provider and c.model == model), None)


@dataclass(frozen=True)
class PaidApproval:
    environment: str
    route: str
    service: str
    policy_version: str
    allowed_provider_accounts: frozenset[str]
    expires_at: datetime
    approval_id: str
    enabled: bool = False

    def __post_init__(self) -> None:
        for label, value in (
            ("environment", self.environment),
            ("route", self.route),
            ("service", self.service),
            ("policy_version", self.policy_version),
            ("approval_id", self.approval_id),
        ):
            validate_identifier(value, label)
        if self.expires_at.tzinfo is None:
            raise ConfigurationError("paid approval expiry must be timezone-aware")
        for value in self.allowed_provider_accounts:
            validate_identifier(value, "allowed provider account")

    def permits(
        self,
        *,
        caller: CallerContext,
        route: str,
        policy_version: str,
        provider_account_alias: str,
        now: datetime,
    ) -> bool:
        return (
            self.enabled
            and self.environment == caller.environment
            and self.route == route
            and self.service == caller.service
            and self.policy_version == policy_version
            and provider_account_alias in self.allowed_provider_accounts
            and now.astimezone(timezone.utc) < self.expires_at.astimezone(timezone.utc)
        )


@dataclass(frozen=True)
class RouterConfig:
    policy_version: str
    routes: Mapping[str, RoutePolicy]
    approvals: tuple[PaidApproval, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.policy_version, "policy_version")
        copied = dict(self.routes)
        if not copied:
            raise ConfigurationError("at least one route is required")
        for name, route in copied.items():
            if name != route.name:
                raise ConfigurationError("route mapping key must match route.name")
        object.__setattr__(self, "routes", MappingProxyType(copied))


@dataclass(frozen=True)
class CandidateHealth:
    provider: str
    model: str
    billing_class: BillingClass
    available: bool
    reason_code: str | None = None


@dataclass(frozen=True)
class RouterHealth:
    healthy: bool
    routes: Mapping[str, tuple[CandidateHealth, ...]]


@dataclass(frozen=True)
class ChargeEstimate:
    micro_usd: int
    input_tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.micro_usd, int) or isinstance(self.micro_usd, bool):
            raise ValueError("charge estimate must use integer micro-USD")
        if self.micro_usd < 0 or self.input_tokens < 0:
            raise ValueError("charge estimate cannot be negative")


@dataclass(frozen=True)
class AdapterRequest:
    request: LLMRequest
    provider: str
    model: str
    provider_account_alias: str
    timeout_seconds: float
    idempotency_key: str | None
    price_card_version: str | None = None


@dataclass(frozen=True)
class AdapterResult:
    output: Any
    usage: Usage | None = None
    provider_request_id: str | None = None
    actual_micro_usd: int | None = None

    def __post_init__(self) -> None:
        if self.actual_micro_usd is not None and (
            not isinstance(self.actual_micro_usd, int)
            or isinstance(self.actual_micro_usd, bool)
            or self.actual_micro_usd < 0
        ):
            raise ValueError("actual charge must use non-negative integer micro-USD")
