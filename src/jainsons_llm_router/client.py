"""Capability-lane client for zero-marginal-cost text completion.

Applications choose a harness capability lane, not a provider or model.  The
client turns the committed harness-derived policy into ordinary
:class:`~jainsons_llm_router.router.Router` routes and deliberately configures
no paid candidates.

Router construction is lazy.  Importing this module does not read credentials,
open files, start a CLI, or make a network request.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import NoReturn

from .adapters import GenericHTTPAdapter, PriceCard
from .errors import (
    ConfigurationError,
    LedgerUnavailable,
    ProviderFailure,
    RouteUnavailable,
)
from .models import (
    BillingClass,
    CallerContext,
    Candidate,
    LLMRequest,
    RoutePolicy,
    RouterConfig,
    UseRoute,
)
from .policies.harness_derived import (
    BACKENDS,
    FREE_CANDIDATE_BACKENDS_BY_LANE,
    HARNESS_POLICY_SHA256,
    free_candidate_backend_order,
    order_free_candidates,
)
from .router import Router, create_router

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_MAX_OUTPUT_TOKENS = 1_024
DEFAULT_TIMEOUT_SECONDS = 30.0

_ENDPOINT_ENV = "JAINSONS_LLM_ROUTER_ENDPOINT"
_API_KEY_ENV_ENV = "JAINSONS_LLM_ROUTER_API_KEY_ENV"
_ENVIRONMENT_ENV = "JAINSONS_LLM_ROUTER_ENVIRONMENT"
_FREE_PRICE_CARD_VERSION = "harness-free-v1"
_ADAPTER_NAME = "capability-http"
_MAX_ROUTE_OUTPUT_TOKENS = 1_000_000

# These lanes need request/response shapes other than text chat.  They must not
# be squeezed through the text adapter merely because their harness backend is
# classified as zero-marginal-cost.
_SPECIAL_CAPABILITY_LANES = frozenset({"image_gen", "vision"})


class _FreeOnlyLedger:
    """Fail closed if a free-only route ever reaches a money-ledger path."""

    spend_domain = "free-only"
    _MESSAGE = "free-only capability client must never perform a ledger operation"

    @classmethod
    def _deny(cls) -> NoReturn:
        raise LedgerUnavailable(cls._MESSAGE)

    def reserve(self, spec: object) -> NoReturn:
        self._deny()

    def settle(self, reservation_id: str, **kwargs: object) -> NoReturn:
        self._deny()

    def release(self, reservation_id: str, **kwargs: object) -> NoReturn:
        self._deny()

    def settle_unknown(self, reservation_id: str, **kwargs: object) -> NoReturn:
        self._deny()

    def remaining_budget(self, scope: str, **kwargs: object) -> NoReturn:
        self._deny()


def lane_primary_backend(lane: str) -> str:
    """Return the primary backend the harness policy assigns to *lane* (e.g. vision -> agy).

    Lets callers resolve a capability's model from the harness-mirrored registry
    instead of a per-repo env var. Falls back to None when unknown.
    """
    order = free_candidate_backend_order(lane)
    return order[0] if order else None


def _lane_candidates(lane: str) -> tuple[Candidate, ...]:
    """Resolve a lane to ordered, explicitly free harness candidates."""

    # This call is intentionally first: it supplies the canonical unknown-lane
    # error from the generated policy instead of maintaining a second lane list.
    backend_order = free_candidate_backend_order(lane)
    if lane in _SPECIAL_CAPABILITY_LANES:
        raise ConfigurationError(
            f"harness lane {lane!r} requires an explicitly configured "
            "capability adapter; complete_text will not make that call"
        )

    by_backend: dict[str, Candidate] = {}
    unavailable: set[str] = set()
    for backend_name in backend_order:
        policy = BACKENDS.get(backend_name)
        if policy is None:
            raise ConfigurationError(
                f"harness lane {lane!r} references unknown backend {backend_name!r}; "
                "regenerate harness_derived.py"
            )
        if policy.billing_class is BillingClass.PAID or policy.funding == "paid_api":
            raise ConfigurationError(
                f"harness lane {lane!r} includes paid-capable backend {backend_name!r}; "
                "explicit paid configuration and a durable ledger are required"
            )
        if not policy.automatic_enabled or not policy.model:
            unavailable.add(backend_name)
            continue
        by_backend[backend_name] = Candidate(
            provider=backend_name,
            model=policy.model,
            adapter=_ADAPTER_NAME,
            billing_class=BillingClass.FREE,
            provider_account_alias=f"{backend_name}-free",
            price_card_version=_FREE_PRICE_CARD_VERSION,
            zero_marginal_cost=True,
            timeout_seconds=3_600.0,
        )

    candidates = order_free_candidates(lane, by_backend, allow_missing=unavailable)
    if not candidates:
        missing = ", ".join(backend_order) or "none"
        raise RouteUnavailable(
            f"harness lane {lane!r} has no configured free HTTP model "
            f"(checked backends: {missing}); sync the harness policy or configure "
            "the capability's explicit adapter"
        )
    return candidates


def _build_router(endpoint: str, api_key_env: str) -> Router:
    routes: dict[str, RoutePolicy] = {}
    models: set[str] = set()
    for lane in FREE_CANDIDATE_BACKENDS_BY_LANE:
        if lane in _SPECIAL_CAPABILITY_LANES:
            continue
        try:
            candidates = _lane_candidates(lane)
        except RouteUnavailable:
            # A lane without an HTTP-addressable free model remains unavailable;
            # asking for that lane later produces the detailed resolver error.
            continue
        routes[lane] = RoutePolicy(
            name=lane,
            free_candidates=candidates,
            paid_candidates=(),
            paid_allowed=False,
            accepted_task_kinds=frozenset({"text_generation"}),
            accepted_modalities=frozenset({"text"}),
            max_output_tokens=_MAX_ROUTE_OUTPUT_TOKENS,
        )
        models.update(candidate.model for candidate in candidates)

    if not routes:
        raise RouteUnavailable(
            "the harness-derived policy contains no HTTP-addressable free text models; "
            "sync the harness policy before using the shared client"
        )

    zero_price = PriceCard(0, 0, fixed_micro_usd=0, rounding_margin_micro_usd=0)
    adapter = GenericHTTPAdapter(
        endpoint=endpoint,
        api_key_env=api_key_env,
        price_cards={_FREE_PRICE_CARD_VERSION: zero_price},
        supported_models=models,
        supported_modalities={"text"},
        verified_free_models={_FREE_PRICE_CARD_VERSION: models},
    )
    config = RouterConfig(
        policy_version=f"harness-{HARNESS_POLICY_SHA256[:16]}",
        routes=routes,
    )
    return create_router(config, adapters={_ADAPTER_NAME: adapter}, ledger=_FreeOnlyLedger())


_ROUTERS: dict[tuple[str, str], Router] = {}
_ROUTERS_LOCK = threading.Lock()


def _get_router(endpoint: str, api_key_env: str) -> Router:
    """Return the lazy singleton for one endpoint/credential pair."""

    cache_key = (endpoint, api_key_env)
    router = _ROUTERS.get(cache_key)
    if router is not None:
        return router
    with _ROUTERS_LOCK:
        router = _ROUTERS.get(cache_key)
        if router is None:
            router = _build_router(endpoint, api_key_env)
            _ROUTERS[cache_key] = router
        return router


def complete_text(
    prompt: str,
    lane: str = "research",
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    service: str = "jainsons-llm-client",
    *,
    endpoint: str | None = None,
    api_key_env: str | None = None,
) -> str:
    """Complete ``prompt`` using the best live free model for ``lane``.

    ``endpoint`` defaults to ``JAINSONS_LLM_ROUTER_ENDPOINT`` and then the
    OpenRouter chat-completions endpoint.  ``api_key_env`` is the *name* of the
    credential environment variable; it defaults through
    ``JAINSONS_LLM_ROUTER_API_KEY_ENV`` to ``OPENROUTER_API_KEY``.  Callers
    never provide a provider or model.

    Paid and non-text capability lanes fail closed and require a separately
    reviewed adapter/ledger configuration.
    """

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")

    # Resolve the requested lane before constructing the shared router.  This
    # keeps unknown and special-capability errors precise and network-free.
    _lane_candidates(lane)

    resolved_endpoint = endpoint or os.environ.get(_ENDPOINT_ENV, DEFAULT_ENDPOINT)
    resolved_key_env = api_key_env or os.environ.get(_API_KEY_ENV_ENV, DEFAULT_API_KEY_ENV)
    router = _get_router(resolved_endpoint, resolved_key_env)
    if lane not in router.config.routes:
        # A generated policy may contain a known lane whose backends currently
        # have no concrete, automatically enabled HTTP model.
        raise RouteUnavailable(
            f"harness lane {lane!r} has no configured free HTTP candidates; "
            "sync the harness policy or configure an explicit capability adapter"
        )

    health = router.health(lane)
    if not health.healthy:
        reasons = sorted(
            {item.reason_code or "unavailable" for item in health.routes[lane] if not item.available}
        )
        reason_text = ", ".join(reasons) or "unavailable"
        raise RouteUnavailable(
            f"harness lane {lane!r} has no live free candidates ({reason_text}); "
            f"ensure {resolved_key_env} is set and the configured endpoint is reachable"
        )

    request = LLMRequest(
        messages=({"role": "user", "content": prompt},),
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    caller = CallerContext(
        service=service,
        environment=os.environ.get(_ENVIRONMENT_ENV, "production"),
        route_purpose=lane,
        deployment_version="capability-client-v1",
        correlation_id=uuid.uuid4().hex,
    )
    try:
        result = router.complete(request, selection=UseRoute(lane), caller=caller)
    except RouteUnavailable as exc:
        raise RouteUnavailable(
            f"no live free candidate succeeded for harness lane {lane!r}; "
            "check endpoint compatibility and refresh the harness-derived policy",
            details=exc.details,
        ) from exc
    if not isinstance(result.output, str):
        raise ProviderFailure("free text provider returned a non-text response")
    return result.output


__all__ = ["complete_text"]
