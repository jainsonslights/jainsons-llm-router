"""Stable public errors raised by the router and ledger."""

from __future__ import annotations

from typing import Any


class RouterError(Exception):
    """Base class for expected router failures.

    ``details`` is deliberately restricted to privacy-safe operational data.
    Prompts, completions, credentials, and raw customer identifiers must never
    be attached to an exception.
    """

    code = "router_error"

    def __init__(self, message: str = "", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message or self.code)
        self.details = dict(details or {})


class RouteUnavailable(RouterError):
    """No configured eligible free route was healthy."""

    code = "route_unavailable"


class PaidNotApproved(RouterError):
    """Paid use is not approved for this caller, route, and environment."""

    code = "paid_not_approved"


class BudgetDenied(RouterError):
    """A call or monetary cap would be exceeded."""

    code = "budget_denied"


class LedgerUnavailable(RouterError):
    """The ledger cannot safely authorize paid dispatch."""

    code = "ledger_unavailable"


class ExactModelUnavailable(RouterError):
    """An exact provider/model selection cannot run."""

    code = "exact_model_unavailable"


class ProviderFailure(RouterError):
    """A provider failed before a definitive successful result."""

    code = "provider_failure"


class OutcomeUnknown(RouterError):
    """A paid request may have reached its provider; do not retry blindly."""

    code = "outcome_unknown"


class InvalidRequest(RouterError):
    """The request violates route or adapter bounds."""

    code = "invalid_request"


class ConfigurationError(RouterError):
    """Router policy is unsafe or internally inconsistent."""

    code = "configuration_error"

