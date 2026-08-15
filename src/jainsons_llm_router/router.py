"""Policy-owned provider routing and paid-dispatch control."""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Mapping, Protocol

from .adapters import AdapterFailure, ProviderAdapter
from .errors import (
    BudgetDenied,
    ConfigurationError,
    ExactModelUnavailable,
    InvalidRequest,
    LedgerUnavailable,
    OutcomeUnknown,
    PaidNotApproved,
    ProviderFailure,
    RouteUnavailable,
)
from .ledger import (
    IdempotencyConflict,
    LedgerReservation,
    ReservationSpec,
    UsageLedger,
    hash_private_identifier,
)
from .models import (
    AdapterRequest,
    AdapterResult,
    AttemptSummary,
    BillingClass,
    BudgetRemaining,
    CallerContext,
    Candidate,
    CandidateHealth,
    ExactModel,
    FailurePhase,
    LLMRequest,
    LLMResult,
    RoutePolicy,
    RouteSelection,
    RouterConfig,
    RouterHealth,
    UseRoute,
    Usage,
)
from .observability import EventLogger, NullEventLogger, stable_hash


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class Router:
    """In-process LLM policy boundary."""

    def __init__(
        self,
        config: RouterConfig,
        *,
        adapters: Mapping[str, ProviderAdapter],
        ledger: UsageLedger,
        logger: EventLogger | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.adapters = dict(adapters)
        self.ledger = ledger
        self.logger = logger or NullEventLogger()
        self.clock = clock or SystemClock()
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        for route in self.config.routes.values():
            automatic_seen: set[tuple[str, str, str]] = set()
            for candidate in route.free_candidates + route.paid_candidates:
                if candidate.adapter not in self.adapters:
                    raise ConfigurationError(f"adapter {candidate.adapter!r} is not configured")
                identity = (candidate.provider, candidate.model, candidate.adapter)
                if identity in automatic_seen:
                    raise ConfigurationError("duplicate automatic route candidate")
                automatic_seen.add(identity)
            exact_seen: set[tuple[str, str, str]] = set()
            for candidate in route.exact_candidates:
                if candidate.adapter not in self.adapters:
                    raise ConfigurationError(f"adapter {candidate.adapter!r} is not configured")
                identity = (candidate.provider, candidate.model, candidate.adapter)
                if identity in exact_seen:
                    raise ConfigurationError("duplicate exact route candidate")
                exact_seen.add(identity)

    def _route(self, selection: RouteSelection, caller: CallerContext) -> RoutePolicy:
        name = selection.name if isinstance(selection, UseRoute) else caller.route_purpose
        route = self.config.routes.get(name)
        if route is None:
            if isinstance(selection, ExactModel):
                raise ExactModelUnavailable("exact selection route is not configured")
            raise RouteUnavailable("route is not configured")
        if route.allowed_callers and caller.identity not in route.allowed_callers:
            raise RouteUnavailable("caller is not authorized for route")
        if route.allowed_environments and caller.environment not in route.allowed_environments:
            raise RouteUnavailable("environment is not authorized for route")
        return route

    @staticmethod
    def _validate_request(route: RoutePolicy, request: LLMRequest) -> None:
        if request.task_kind not in route.accepted_task_kinds:
            raise InvalidRequest("task kind is not accepted by route")
        if request.modality not in route.accepted_modalities:
            raise InvalidRequest("modality is not accepted by route")
        if request.input_size > route.max_input_size:
            raise InvalidRequest("input exceeds route bound")
        if request.max_output_tokens > route.max_output_tokens:
            raise InvalidRequest("output token bound exceeds route policy")

    def _free_is_eligible(self, candidate: Candidate, adapter: ProviderAdapter) -> tuple[bool, str | None]:
        if not candidate.zero_marginal_cost:
            return False, "zero_cost_not_proven"
        try:
            if not adapter.is_non_billable(
                model=candidate.model, price_card_version=candidate.price_card_version
            ):
                return False, "zero_price_not_verified"
            healthy, reason = adapter.health()
        except Exception:
            return False, "adapter_health_failed"
        return (healthy, reason if not healthy else None)

    def _approval(self, route: RoutePolicy, candidate: Candidate, caller: CallerContext) -> bool:
        if not route.paid_allowed:
            return False
        now = self.clock.now()
        if now.tzinfo is None:
            raise ConfigurationError("clock must return a timezone-aware datetime")
        return any(
            approval.permits(
                caller=caller,
                route=route.name,
                policy_version=self.config.policy_version,
                provider_account_alias=candidate.provider_account_alias,
                now=now,
            )
            for approval in self.config.approvals
        )

    def _emit(
        self,
        *,
        route: RoutePolicy,
        caller: CallerContext,
        candidate: Candidate,
        outcome: str,
        reason_code: str | None,
        attempt_number: int,
        latency_ms: int = 0,
        reservation_id: str | None = None,
        provider_request_id: str | None = None,
        approval: bool | None = None,
    ) -> None:
        self.logger.emit(
            {
                "event_type": "router_attempt",
                "route": route.name,
                "provider": candidate.provider,
                "model": candidate.model,
                "billing_class": candidate.billing_class.value,
                "approval": approval,
                "cap_scopes": list(candidate.scopes) if candidate.billing_class is BillingClass.PAID else [],
                "reservation_id": reservation_id,
                "latency_ms": latency_ms,
                "outcome": outcome,
                "reason_code": reason_code,
                "attempt_number": attempt_number,
                "correlation_id_hash": stable_hash(caller.correlation_id),
                "provider_request_id_hash": stable_hash(provider_request_id),
                "provider_account_alias_hash": stable_hash(candidate.provider_account_alias),
                "service": caller.service,
                "environment": caller.environment,
                "policy_version": self.config.policy_version,
            }
        )

    @staticmethod
    def _adapter_request(request: LLMRequest, candidate: Candidate) -> AdapterRequest:
        timeout = min(float(request.timeout_seconds), float(candidate.timeout_seconds))
        return AdapterRequest(
            request=request,
            provider=candidate.provider,
            model=candidate.model,
            provider_account_alias=candidate.provider_account_alias,
            timeout_seconds=timeout,
            idempotency_key=request.idempotency_key,
            price_card_version=candidate.price_card_version,
        )

    def _reserve(
        self,
        request: LLMRequest,
        route: RoutePolicy,
        caller: CallerContext,
        candidate: Candidate,
        adapter: ProviderAdapter,
        *,
        attempt_number: int,
    ) -> LedgerReservation:
        if not request.idempotency_key:
            raise InvalidRequest("a stable idempotency_key is required before paid dispatch")
        try:
            estimate = adapter.estimate_max_charge(
                request,
                model=candidate.model,
                price_card_version=candidate.price_card_version or "",
            )
        except AdapterFailure as exc:
            raise ProviderFailure("adapter cannot bound paid request", details={"reason_code": exc.reason_code}) from exc
        except Exception as exc:
            raise ProviderFailure("adapter cannot bound paid request", details={"reason_code": "adapter_contract_error"}) from exc
        if estimate.micro_usd <= 0:
            raise ProviderFailure("paid adapter returned a non-positive reservation")
        spec = ReservationSpec(
            spend_domain=self.ledger.spend_domain,
            aggregate_scope=candidate.aggregate_scope,
            provider_scope=candidate.provider_scope,
            route_scope=candidate.route_scope,
            model_scope=candidate.model_scope,
            provider_account_alias=candidate.provider_account_alias,
            provider=candidate.provider,
            model=candidate.model,
            service=caller.service,
            environment=caller.environment,
            route=route.name,
            policy_version=self.config.policy_version,
            caller_id=caller.identity,
            correlation_id_hash=hash_private_identifier(caller.correlation_id),
            idempotency_key_hash=hash_private_identifier(request.idempotency_key),
            reserved_micro_usd=estimate.micro_usd,
            price_card_version=candidate.price_card_version or "",
            attempt_number=attempt_number,
        )
        try:
            return self.ledger.reserve(spec)
        except IdempotencyConflict as exc:
            if exc.status in {"live", "unknown"}:
                raise OutcomeUnknown("paid idempotency key is unresolved") from exc
            raise BudgetDenied("paid idempotency key has already been consumed") from exc

    def _successful_result(
        self,
        result: AdapterResult,
        candidate: Candidate,
        attempts: list[AttemptSummary],
        *,
        reservation_id: str | None,
        elapsed_ms: int,
    ) -> LLMResult:
        return LLMResult(
            output=result.output,
            provider=candidate.provider,
            model=candidate.model,
            usage=result.usage,
            provider_request_id=result.provider_request_id,
            billing_class=candidate.billing_class,
            reservation_id=reservation_id,
            elapsed_ms=elapsed_ms,
            attempts=tuple(attempts),
        )

    def _handle_paid_failure(
        self,
        *,
        exc: AdapterFailure,
        reservation: LedgerReservation,
    ) -> None:
        provider_id_hash = stable_hash(exc.provider_request_id)
        if exc.phase is FailurePhase.PRE_DISPATCH:
            self.ledger.release(reservation.reservation_id, reason=exc.reason_code)
        elif exc.phase is FailurePhase.DEFINITIVE:
            self.ledger.settle(
                reservation.reservation_id,
                actual_micro_usd=exc.actual_micro_usd or 0,
                provider_request_id_hash=provider_id_hash,
                outcome="provider_failure",
            )
        else:
            self.ledger.settle_unknown(
                reservation.reservation_id,
                reason=exc.reason_code,
                provider_request_id_hash=provider_id_hash,
            )

    def _settle_success(
        self,
        result: AdapterResult,
        reservation: LedgerReservation,
    ) -> None:
        actual = result.actual_micro_usd
        if actual is None:
            actual = reservation.reserved_micro_usd
        usage = result.usage or Usage()
        self.ledger.settle(
            reservation.reservation_id,
            actual_micro_usd=actual,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            provider_request_id_hash=stable_hash(result.provider_request_id),
        )

    def _run_free_sync(
        self,
        request: LLMRequest,
        route: RoutePolicy,
        caller: CallerContext,
        candidate: Candidate,
        attempts: list[AttemptSummary],
        *,
        exact: bool,
    ) -> LLMResult | None:
        adapter = self.adapters[candidate.adapter]
        eligible, reason = self._free_is_eligible(candidate, adapter)
        if not eligible:
            self._emit(
                route=route, caller=caller, candidate=candidate, outcome="skipped",
                reason_code=reason, attempt_number=1,
            )
            if exact:
                raise ExactModelUnavailable("exact free model is unavailable")
            return None
        max_attempts = 1 if exact else candidate.retry_attempts
        for number in range(1, max_attempts + 1):
            started = time.monotonic()
            try:
                adapter.validate_capability(
                    request,
                    model=candidate.model,
                    timeout_seconds=min(request.timeout_seconds, candidate.timeout_seconds),
                )
                result = adapter.complete(self._adapter_request(request, candidate))
            except AdapterFailure as exc:
                elapsed = int((time.monotonic() - started) * 1000)
                attempts.append(
                    AttemptSummary(candidate.provider, candidate.model, candidate.billing_class, "failure", exc.reason_code, elapsed)
                )
                self._emit(
                    route=route, caller=caller, candidate=candidate, outcome="failure",
                    reason_code=exc.reason_code, attempt_number=number, latency_ms=elapsed,
                )
                if exact:
                    raise ProviderFailure("exact provider failed", details={"reason_code": exc.reason_code}) from exc
                continue
            except Exception as exc:
                elapsed = int((time.monotonic() - started) * 1000)
                attempts.append(
                    AttemptSummary(candidate.provider, candidate.model, candidate.billing_class, "failure", "adapter_contract_error", elapsed)
                )
                self._emit(
                    route=route, caller=caller, candidate=candidate, outcome="failure",
                    reason_code="adapter_contract_error", attempt_number=number, latency_ms=elapsed,
                )
                if exact:
                    raise ProviderFailure("exact provider adapter contract failed") from exc
                continue
            elapsed = int((time.monotonic() - started) * 1000)
            attempts.append(AttemptSummary(candidate.provider, candidate.model, candidate.billing_class, "success", None, elapsed))
            self._emit(
                route=route, caller=caller, candidate=candidate, outcome="success", reason_code=None,
                attempt_number=number, latency_ms=elapsed, provider_request_id=result.provider_request_id,
            )
            return self._successful_result(result, candidate, attempts, reservation_id=None, elapsed_ms=elapsed)
        return None

    def _run_paid_sync(
        self,
        request: LLMRequest,
        route: RoutePolicy,
        caller: CallerContext,
        candidate: Candidate,
        attempts: list[AttemptSummary],
        *,
        exact: bool,
        attempt_number: int,
    ) -> LLMResult | None:
        approved = self._approval(route, candidate, caller)
        if not approved:
            self._emit(
                route=route, caller=caller, candidate=candidate, outcome="denied",
                reason_code="paid_not_approved", attempt_number=attempt_number, approval=False,
            )
            if exact:
                raise PaidNotApproved("exact paid model is not approved")
            return None
        adapter = self.adapters[candidate.adapter]
        try:
            adapter.validate_capability(
                request,
                model=candidate.model,
                timeout_seconds=min(request.timeout_seconds, candidate.timeout_seconds),
            )
        except AdapterFailure as exc:
            self._emit(
                route=route, caller=caller, candidate=candidate, outcome="failure",
                reason_code=exc.reason_code, attempt_number=attempt_number, approval=True,
            )
            if exact:
                raise ProviderFailure("exact paid provider is unavailable", details={"reason_code": exc.reason_code}) from exc
            return None
        except Exception as exc:
            self._emit(
                route=route, caller=caller, candidate=candidate, outcome="failure",
                reason_code="adapter_contract_error", attempt_number=attempt_number, approval=True,
            )
            if exact:
                raise ProviderFailure("exact paid provider adapter contract failed") from exc
            return None
        reservation = self._reserve(
            request, route, caller, candidate, adapter, attempt_number=attempt_number
        )
        started = time.monotonic()
        try:
            result = adapter.complete(self._adapter_request(request, candidate))
        except AdapterFailure as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            try:
                self._handle_paid_failure(exc=exc, reservation=reservation)
            except LedgerUnavailable as ledger_exc:
                raise OutcomeUnknown("ledger failed while recording paid provider outcome") from ledger_exc
            attempts.append(
                AttemptSummary(candidate.provider, candidate.model, candidate.billing_class, "failure", exc.reason_code, elapsed)
            )
            self._emit(
                route=route, caller=caller, candidate=candidate,
                outcome="unknown" if exc.phase is FailurePhase.UNKNOWN else "failure",
                reason_code=exc.reason_code, attempt_number=attempt_number, latency_ms=elapsed,
                reservation_id=reservation.reservation_id, provider_request_id=exc.provider_request_id,
                approval=True,
            )
            if exc.phase is FailurePhase.UNKNOWN:
                raise OutcomeUnknown("paid provider outcome is unknown") from exc
            if exact:
                raise ProviderFailure("exact paid provider failed", details={"reason_code": exc.reason_code}) from exc
            return None
        except Exception as exc:
            # A non-normalized exception after complete() begins cannot prove
            # pre-dispatch status, so it is conservatively unknown.
            elapsed = int((time.monotonic() - started) * 1000)
            try:
                self.ledger.settle_unknown(reservation.reservation_id, reason="adapter_contract_error")
            except LedgerUnavailable:
                pass
            self._emit(
                route=route, caller=caller, candidate=candidate, outcome="unknown",
                reason_code="adapter_contract_error", attempt_number=attempt_number,
                latency_ms=elapsed, reservation_id=reservation.reservation_id, approval=True,
            )
            raise OutcomeUnknown("adapter failed after paid dispatch boundary") from exc
        elapsed = int((time.monotonic() - started) * 1000)
        try:
            self._settle_success(result, reservation)
        except LedgerUnavailable as exc:
            raise OutcomeUnknown("ledger failed while settling paid success") from exc
        attempts.append(AttemptSummary(candidate.provider, candidate.model, candidate.billing_class, "success", None, elapsed))
        self._emit(
            route=route, caller=caller, candidate=candidate, outcome="success", reason_code=None,
            attempt_number=attempt_number, latency_ms=elapsed, reservation_id=reservation.reservation_id,
            provider_request_id=result.provider_request_id, approval=True,
        )
        return self._successful_result(
            result, candidate, attempts, reservation_id=reservation.reservation_id, elapsed_ms=elapsed
        )

    def complete(
        self,
        request: LLMRequest,
        *,
        selection: RouteSelection = UseRoute("default"),
        caller: CallerContext,
    ) -> LLMResult:
        route = self._route(selection, caller)
        self._validate_request(route, request)
        attempts: list[AttemptSummary] = []
        if isinstance(selection, ExactModel):
            candidate = route.find_exact(selection.provider, selection.model)
            if candidate is None:
                raise ExactModelUnavailable("provider/model is not in route allow-list")
            if candidate.billing_class is BillingClass.FREE:
                result = self._run_free_sync(request, route, caller, candidate, attempts, exact=True)
            else:
                result = self._run_paid_sync(
                    request, route, caller, candidate, attempts, exact=True, attempt_number=1
                )
            if result is None:  # defensive; exact helpers raise on failure.
                raise ExactModelUnavailable("exact provider/model could not run")
            return result

        free_candidates = route.free_candidates
        paid_candidates = route.paid_candidates
        if not route.substitutions_allowed:
            combined = free_candidates + paid_candidates
            free_candidates = combined[:1] if combined and combined[0].billing_class is BillingClass.FREE else ()
            paid_candidates = combined[:1] if combined and combined[0].billing_class is BillingClass.PAID else ()
        for candidate in free_candidates:
            result = self._run_free_sync(request, route, caller, candidate, attempts, exact=False)
            if result is not None:
                return result
        if not paid_candidates:
            raise RouteUnavailable("no configured eligible route succeeded", details={"attempts": len(attempts)})

        any_approved = False
        last_budget_error: BudgetDenied | None = None
        for number, candidate in enumerate(paid_candidates, start=1):
            any_approved = any_approved or self._approval(route, candidate, caller)
            try:
                result = self._run_paid_sync(
                    request, route, caller, candidate, attempts, exact=False, attempt_number=number
                )
            except BudgetDenied as exc:
                last_budget_error = exc
                self._emit(
                    route=route, caller=caller, candidate=candidate, outcome="denied",
                    reason_code=exc.code, attempt_number=number, approval=True,
                )
                continue
            if result is not None:
                return result
        if last_budget_error is not None:
            raise last_budget_error
        if not any_approved:
            raise PaidNotApproved("paid tier is disabled or approval is invalid")
        raise ProviderFailure("all explicitly configured paid providers failed", details={"attempts": len(attempts)})

    async def _run_free_async(
        self,
        request: LLMRequest,
        route: RoutePolicy,
        caller: CallerContext,
        candidate: Candidate,
        attempts: list[AttemptSummary],
        *,
        exact: bool,
    ) -> LLMResult | None:
        adapter = self.adapters[candidate.adapter]
        eligible, reason = self._free_is_eligible(candidate, adapter)
        if not eligible:
            self._emit(
                route=route, caller=caller, candidate=candidate, outcome="skipped",
                reason_code=reason, attempt_number=1,
            )
            if exact:
                raise ExactModelUnavailable("exact free model is unavailable")
            return None
        max_attempts = 1 if exact else candidate.retry_attempts
        for number in range(1, max_attempts + 1):
            started = time.monotonic()
            try:
                adapter.validate_capability(
                    request, model=candidate.model,
                    timeout_seconds=min(request.timeout_seconds, candidate.timeout_seconds),
                )
                result = await adapter.acomplete(self._adapter_request(request, candidate))
            except AdapterFailure as exc:
                elapsed = int((time.monotonic() - started) * 1000)
                attempts.append(AttemptSummary(candidate.provider, candidate.model, candidate.billing_class, "failure", exc.reason_code, elapsed))
                self._emit(
                    route=route, caller=caller, candidate=candidate, outcome="failure",
                    reason_code=exc.reason_code, attempt_number=number, latency_ms=elapsed,
                )
                if exact:
                    raise ProviderFailure("exact provider failed", details={"reason_code": exc.reason_code}) from exc
                continue
            except Exception as exc:
                elapsed = int((time.monotonic() - started) * 1000)
                attempts.append(AttemptSummary(candidate.provider, candidate.model, candidate.billing_class, "failure", "adapter_contract_error", elapsed))
                self._emit(
                    route=route, caller=caller, candidate=candidate, outcome="failure",
                    reason_code="adapter_contract_error", attempt_number=number, latency_ms=elapsed,
                )
                if exact:
                    raise ProviderFailure("exact provider adapter contract failed") from exc
                continue
            elapsed = int((time.monotonic() - started) * 1000)
            attempts.append(AttemptSummary(candidate.provider, candidate.model, candidate.billing_class, "success", None, elapsed))
            self._emit(
                route=route, caller=caller, candidate=candidate, outcome="success", reason_code=None,
                attempt_number=number, latency_ms=elapsed, provider_request_id=result.provider_request_id,
            )
            return self._successful_result(result, candidate, attempts, reservation_id=None, elapsed_ms=elapsed)
        return None

    async def _run_paid_async(
        self,
        request: LLMRequest,
        route: RoutePolicy,
        caller: CallerContext,
        candidate: Candidate,
        attempts: list[AttemptSummary],
        *,
        exact: bool,
        attempt_number: int,
    ) -> LLMResult | None:
        approved = self._approval(route, candidate, caller)
        if not approved:
            self._emit(
                route=route, caller=caller, candidate=candidate, outcome="denied",
                reason_code="paid_not_approved", attempt_number=attempt_number, approval=False,
            )
            if exact:
                raise PaidNotApproved("exact paid model is not approved")
            return None
        adapter = self.adapters[candidate.adapter]
        try:
            adapter.validate_capability(
                request, model=candidate.model,
                timeout_seconds=min(request.timeout_seconds, candidate.timeout_seconds),
            )
        except AdapterFailure as exc:
            if exact:
                raise ProviderFailure("exact paid provider is unavailable", details={"reason_code": exc.reason_code}) from exc
            return None
        except Exception as exc:
            self._emit(
                route=route, caller=caller, candidate=candidate, outcome="failure",
                reason_code="adapter_contract_error", attempt_number=attempt_number, approval=True,
            )
            if exact:
                raise ProviderFailure("exact paid provider adapter contract failed") from exc
            return None
        reservation = self._reserve(
            request, route, caller, candidate, adapter, attempt_number=attempt_number
        )
        started = time.monotonic()
        try:
            result = await adapter.acomplete(self._adapter_request(request, candidate))
        except AdapterFailure as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            try:
                self._handle_paid_failure(exc=exc, reservation=reservation)
            except LedgerUnavailable as ledger_exc:
                raise OutcomeUnknown("ledger failed while recording paid provider outcome") from ledger_exc
            attempts.append(AttemptSummary(candidate.provider, candidate.model, candidate.billing_class, "failure", exc.reason_code, elapsed))
            self._emit(
                route=route, caller=caller, candidate=candidate,
                outcome="unknown" if exc.phase is FailurePhase.UNKNOWN else "failure",
                reason_code=exc.reason_code, attempt_number=attempt_number, latency_ms=elapsed,
                reservation_id=reservation.reservation_id, provider_request_id=exc.provider_request_id,
                approval=True,
            )
            if exc.phase is FailurePhase.UNKNOWN:
                raise OutcomeUnknown("paid provider outcome is unknown") from exc
            if exact:
                raise ProviderFailure("exact paid provider failed", details={"reason_code": exc.reason_code}) from exc
            return None
        except Exception as exc:
            try:
                self.ledger.settle_unknown(reservation.reservation_id, reason="adapter_contract_error")
            except LedgerUnavailable:
                pass
            raise OutcomeUnknown("adapter failed after paid dispatch boundary") from exc
        elapsed = int((time.monotonic() - started) * 1000)
        try:
            self._settle_success(result, reservation)
        except LedgerUnavailable as exc:
            raise OutcomeUnknown("ledger failed while settling paid success") from exc
        attempts.append(AttemptSummary(candidate.provider, candidate.model, candidate.billing_class, "success", None, elapsed))
        self._emit(
            route=route, caller=caller, candidate=candidate, outcome="success", reason_code=None,
            attempt_number=attempt_number, latency_ms=elapsed, reservation_id=reservation.reservation_id,
            provider_request_id=result.provider_request_id, approval=True,
        )
        return self._successful_result(result, candidate, attempts, reservation_id=reservation.reservation_id, elapsed_ms=elapsed)

    async def acomplete(
        self,
        request: LLMRequest,
        *,
        selection: RouteSelection = UseRoute("default"),
        caller: CallerContext,
    ) -> LLMResult:
        route = self._route(selection, caller)
        self._validate_request(route, request)
        attempts: list[AttemptSummary] = []
        if isinstance(selection, ExactModel):
            candidate = route.find_exact(selection.provider, selection.model)
            if candidate is None:
                raise ExactModelUnavailable("provider/model is not in route allow-list")
            if candidate.billing_class is BillingClass.FREE:
                result = await self._run_free_async(request, route, caller, candidate, attempts, exact=True)
            else:
                result = await self._run_paid_async(
                    request, route, caller, candidate, attempts, exact=True, attempt_number=1
                )
            if result is None:
                raise ExactModelUnavailable("exact provider/model could not run")
            return result

        free_candidates = route.free_candidates
        paid_candidates = route.paid_candidates
        if not route.substitutions_allowed:
            combined = free_candidates + paid_candidates
            free_candidates = combined[:1] if combined and combined[0].billing_class is BillingClass.FREE else ()
            paid_candidates = combined[:1] if combined and combined[0].billing_class is BillingClass.PAID else ()
        for candidate in free_candidates:
            result = await self._run_free_async(request, route, caller, candidate, attempts, exact=False)
            if result is not None:
                return result
        if not paid_candidates:
            raise RouteUnavailable("no configured eligible route succeeded", details={"attempts": len(attempts)})
        any_approved = False
        last_budget_error: BudgetDenied | None = None
        for number, candidate in enumerate(paid_candidates, start=1):
            any_approved = any_approved or self._approval(route, candidate, caller)
            try:
                result = await self._run_paid_async(
                    request, route, caller, candidate, attempts, exact=False, attempt_number=number
                )
            except BudgetDenied as exc:
                last_budget_error = exc
                continue
            if result is not None:
                return result
        if last_budget_error is not None:
            raise last_budget_error
        if not any_approved:
            raise PaidNotApproved("paid tier is disabled or approval is invalid")
        raise ProviderFailure("all explicitly configured paid providers failed", details={"attempts": len(attempts)})

    def health(self, route: str | None = None) -> RouterHealth:
        if route is not None and route not in self.config.routes:
            raise RouteUnavailable("route is not configured")
        selected = {route: self.config.routes[route]} if route is not None else dict(self.config.routes)
        report: dict[str, tuple[CandidateHealth, ...]] = {}
        overall = True
        for name, policy in selected.items():
            candidates: list[CandidateHealth] = []
            for candidate in policy.free_candidates + policy.paid_candidates:
                adapter = self.adapters[candidate.adapter]
                try:
                    available, reason = adapter.health()
                    if candidate.billing_class is BillingClass.FREE:
                        proven, proof_reason = self._free_is_eligible(candidate, adapter)
                        available = available and proven
                        reason = reason or proof_reason
                except Exception:
                    available, reason = False, "adapter_health_failed"
                candidates.append(
                    CandidateHealth(candidate.provider, candidate.model, candidate.billing_class, available, reason)
                )
            report[name] = tuple(candidates)
            overall = overall and any(item.available for item in candidates)
        return RouterHealth(healthy=overall, routes=report)

    def remaining_budget(self, scope: str, *, caller: CallerContext) -> BudgetRemaining:
        # Bind budget introspection to a route-authorized caller as well.
        self._route(UseRoute(caller.route_purpose), caller)
        return self.ledger.remaining_budget(scope, caller=caller)


def create_router(
    config: RouterConfig,
    *,
    adapters: Mapping[str, ProviderAdapter],
    ledger: UsageLedger,
    logger: EventLogger | None = None,
    clock: Clock | None = None,
) -> Router:
    return Router(config, adapters=adapters, ledger=ledger, logger=logger, clock=clock)
