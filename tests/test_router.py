from __future__ import annotations

import asyncio
import json

import pytest

from jainsons_llm_router import (
    AdapterFailure,
    AdapterResult,
    BillingClass,
    BudgetCap,
    BudgetDenied,
    Candidate,
    ConfigurationError,
    ExactModel,
    FakeAdapter,
    FailurePhase,
    InMemoryEventLogger,
    LedgerUnavailable,
    LLMRequest,
    OutcomeUnknown,
    PaidNotApproved,
    ProviderFailure,
    RoutePolicy,
    RouterConfig,
    RouteUnavailable,
    UseRoute,
    create_router,
)

from conftest import approval, free_candidate, make_ledger, paid_candidate


def router_for(tmp_path, route, adapters, *, approvals=(), caps=None, logger=None):
    ledger = make_ledger(tmp_path, caps=caps)
    return create_router(
        RouterConfig(policy_version="policy-v1", routes={route.name: route}, approvals=tuple(approvals)),
        adapters=adapters,
        ledger=ledger,
        logger=logger,
    )


def test_route_order_is_configured_paid_needs_approval_and_glm_cannot_be_automatic(
    tmp_path, caller, llm_request
):
    first = free_candidate("first")
    second = free_candidate("second")
    paid = paid_candidate("paid")
    first_adapter = FakeAdapter(
        [AdapterFailure("first_failed", phase=FailurePhase.PRE_DISPATCH)],
        nonbillable_models={first.model},
    )
    second_adapter = FakeAdapter(nonbillable_models={second.model})
    paid_adapter = FakeAdapter()
    route = RoutePolicy(
        name="default", free_candidates=(first, second), paid_candidates=(paid,), paid_allowed=True
    )
    router = router_for(
        tmp_path,
        route,
        {"first": first_adapter, "second": second_adapter, "paid": paid_adapter},
    )

    result = router.complete(llm_request, caller=caller)
    assert result.provider == "second"
    assert [call.provider for call in first_adapter.calls + second_adapter.calls] == ["first", "second"]
    assert paid_adapter.calls == []

    paid_only = RoutePolicy(name="paid-only", paid_candidates=(paid,), paid_allowed=True)
    router = router_for(
        tmp_path / "paid-only",
        paid_only,
        {"paid": paid_adapter},
    )
    paid_caller = type(caller)(
        service=caller.service,
        environment=caller.environment,
        route_purpose="paid-only",
        deployment_version=caller.deployment_version,
        correlation_id="correlation-paid-only",
        caller_id=caller.caller_id,
    )
    with pytest.raises(PaidNotApproved):
        router.complete(llm_request, selection=UseRoute("paid-only"), caller=paid_caller)
    assert paid_adapter.calls == []

    with pytest.raises(ConfigurationError, match="GLM"):
        RoutePolicy(name="bad", free_candidates=(free_candidate("not-glm", model="glm-4"),))
    with pytest.raises(ConfigurationError, match="GLM"):
        RoutePolicy(name="bad2", paid_candidates=(paid_candidate("glm"),))


@pytest.mark.parametrize("failure", ["provider", "approval", "budget"])
def test_exact_model_never_substitutes_on_failure_denial_or_cap(
    tmp_path, caller, llm_request, failure
):
    gemini = paid_candidate("gemini")
    anthropic = paid_candidate("anthropic")
    route = RoutePolicy(
        name="default", paid_candidates=(gemini, anthropic), paid_allowed=failure != "approval"
    )
    gemini_adapter = FakeAdapter(
        [AdapterFailure("boom", phase=FailurePhase.PRE_DISPATCH)] if failure == "provider" else []
    )
    anthropic_adapter = FakeAdapter()
    caps = None
    if failure == "budget":
        caps = {
            "agg": BudgetCap(10, 99),
            "route": BudgetCap(10, 99),
            "provider:gemini": BudgetCap(10, 99),
            "provider:anthropic": BudgetCap(10, 99),
        }
    router = router_for(
        tmp_path,
        route,
        {"gemini": gemini_adapter, "anthropic": anthropic_adapter},
        approvals=(approval(gemini.provider_account_alias, anthropic.provider_account_alias),),
        caps=caps,
    )

    expected = {
        "provider": ProviderFailure,
        "approval": PaidNotApproved,
        "budget": BudgetDenied,
    }[failure]
    with pytest.raises(expected):
        router.complete(llm_request, selection=ExactModel("gemini", gemini.model), caller=caller)
    assert len(gemini_adapter.calls) == (1 if failure == "provider" else 0)
    assert anthropic_adapter.calls == []


def test_paid_siblings_share_aggregate_and_have_own_provider_scopes(tmp_path, caller, llm_request):
    gemini = paid_candidate("gemini")
    anthropic = paid_candidate("anthropic")
    route = RoutePolicy(name="default", paid_candidates=(gemini, anthropic), paid_allowed=True)
    gemini_adapter = FakeAdapter(
        [AdapterFailure("rejected", phase=FailurePhase.DEFINITIVE, actual_micro_usd=0)]
    )
    anthropic_adapter = FakeAdapter()
    router = router_for(
        tmp_path,
        route,
        {"gemini": gemini_adapter, "anthropic": anthropic_adapter},
        approvals=(approval(gemini.provider_account_alias, anthropic.provider_account_alias),),
    )

    result = router.complete(llm_request, caller=caller)
    assert result.provider == "anthropic"
    events = [json.loads(line) for line in router.ledger.journal_path.read_text().splitlines()]
    reserves = [event for event in events if event["event_type"] == "RESERVE"]
    assert [event["aggregate_scope"] for event in reserves] == ["agg", "agg"]
    assert {event["provider_scope"] for event in reserves} == {
        "provider:gemini",
        "provider:anthropic",
    }
    assert router.remaining_budget("agg", caller=caller).calls_remaining == 18
    assert router.remaining_budget("provider:gemini", caller=caller).calls_remaining == 19
    assert router.remaining_budget("provider:anthropic", caller=caller).calls_remaining == 19


def test_crash_boundaries_release_before_dispatch_and_charge_timeout(tmp_path, caller, llm_request):
    paid = paid_candidate("paid")
    route = RoutePolicy(name="default", paid_candidates=(paid,), paid_allowed=True)
    pre = FakeAdapter([AdapterFailure("pre", phase=FailurePhase.PRE_DISPATCH)])
    router = router_for(
        tmp_path / "pre",
        route,
        {"paid": pre},
        approvals=(approval(paid.provider_account_alias),),
    )
    with pytest.raises(ProviderFailure):
        router.complete(llm_request, selection=ExactModel("paid", paid.model), caller=caller)
    events = [json.loads(line) for line in router.ledger.journal_path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["RESERVE", "RELEASE"]
    assert router.remaining_budget("agg", caller=caller).calls_remaining == 20

    unknown = FakeAdapter([AdapterFailure("timeout", phase=FailurePhase.UNKNOWN)])
    router = router_for(
        tmp_path / "unknown",
        route,
        {"paid": unknown},
        approvals=(approval(paid.provider_account_alias),),
    )
    with pytest.raises(OutcomeUnknown):
        router.complete(llm_request, selection=ExactModel("paid", paid.model), caller=caller)
    events = [json.loads(line) for line in router.ledger.journal_path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["RESERVE", "SETTLE_UNKNOWN"]
    assert router.remaining_budget("agg", caller=caller).calls_remaining == 19


def test_duplicate_paid_idempotency_key_cannot_reserve_or_dispatch_twice(tmp_path, caller, llm_request):
    paid = paid_candidate("paid")
    route = RoutePolicy(name="default", paid_candidates=(paid,), paid_allowed=True)
    adapter = FakeAdapter([AdapterFailure("timeout", phase=FailurePhase.UNKNOWN)])
    router = router_for(
        tmp_path,
        route,
        {"paid": adapter},
        approvals=(approval(paid.provider_account_alias),),
    )
    with pytest.raises(OutcomeUnknown):
        router.complete(llm_request, selection=ExactModel("paid", paid.model), caller=caller)
    with pytest.raises(OutcomeUnknown):
        router.complete(llm_request, selection=ExactModel("paid", paid.model), caller=caller)
    assert len(adapter.calls) == 1
    events = [json.loads(line) for line in router.ledger.journal_path.read_text().splitlines()]
    assert [event["event_type"] for event in events].count("RESERVE") == 1


def test_ledger_failure_denies_paid_but_does_not_stop_free_route(tmp_path, caller, llm_request):
    free = free_candidate("free")
    paid = paid_candidate("paid")
    route = RoutePolicy(
        name="default", free_candidates=(free,), paid_candidates=(paid,), paid_allowed=True
    )
    free_adapter = FakeAdapter(nonbillable_models={free.model})
    paid_adapter = FakeAdapter()
    router = router_for(
        tmp_path,
        route,
        {"free": free_adapter, "paid": paid_adapter},
        approvals=(approval(paid.provider_account_alias),),
    )
    router.ledger.journal_path.write_bytes(b'{"partial"')
    assert router.complete(llm_request, caller=caller).provider == "free"
    assert paid_adapter.calls == []

    paid_only = RoutePolicy(name="paid-only", paid_candidates=(paid,), paid_allowed=True)
    paid_router = router_for(
        tmp_path / "only",
        paid_only,
        {"paid": paid_adapter},
        approvals=(approval(paid.provider_account_alias, route="paid-only"),),
    )
    paid_router.ledger.snapshot_path.write_text("not-json", encoding="utf-8")
    paid_caller = type(caller)(
        service=caller.service,
        environment=caller.environment,
        route_purpose="paid-only",
        deployment_version=caller.deployment_version,
        correlation_id="corr-paid-only",
        caller_id=caller.caller_id,
    )
    with pytest.raises(LedgerUnavailable):
        paid_router.complete(llm_request, selection=UseRoute("paid-only"), caller=paid_caller)


@pytest.mark.parametrize("fault", ["partial_journal", "lock_permissions", "unknown_price_card"])
def test_each_paid_integrity_failure_still_allows_a_free_route(
    tmp_path, caller, llm_request, fault
):
    free = free_candidate("free")
    paid = paid_candidate(
        "paid", price_card_version="unknown-prices" if fault == "unknown_price_card" else "prices-v1"
    )
    route = RoutePolicy(
        name="default", free_candidates=(free,), paid_candidates=(paid,), paid_allowed=True
    )
    free_adapter = FakeAdapter(nonbillable_models={free.model})
    paid_adapter = FakeAdapter()
    router = router_for(
        tmp_path,
        route,
        {"free": free_adapter, "paid": paid_adapter},
        approvals=(approval(paid.provider_account_alias),),
    )
    if fault == "partial_journal":
        router.ledger.journal_path.write_bytes(b'{"incomplete"')
    elif fault == "lock_permissions":
        router.ledger.lock_path.chmod(0)

    assert router.complete(llm_request, caller=caller).provider == "free"
    with pytest.raises(LedgerUnavailable):
        router.complete(llm_request, selection=ExactModel("paid", paid.model), caller=caller)
    assert paid_adapter.calls == []


def test_sync_and_async_free_calls_have_equivalent_policy_decision(tmp_path, caller, llm_request):
    free = free_candidate("free")
    route = RoutePolicy(name="default", free_candidates=(free,))
    sync_adapter = FakeAdapter(nonbillable_models={free.model})
    async_adapter = FakeAdapter(nonbillable_models={free.model})
    sync_router = router_for(tmp_path / "sync", route, {"free": sync_adapter})
    async_router = router_for(tmp_path / "async", route, {"free": async_adapter})

    sync_result = sync_router.complete(llm_request, caller=caller)
    async_result = asyncio.run(async_router.acomplete(llm_request, caller=caller))
    assert (sync_result.provider, sync_result.model, sync_result.paid) == (
        async_result.provider,
        async_result.model,
        async_result.paid,
    )


def test_health_and_no_free_route_error(tmp_path, caller, llm_request):
    unavailable = free_candidate("free")
    route = RoutePolicy(name="default", free_candidates=(unavailable,))
    router = router_for(tmp_path, route, {"free": FakeAdapter(healthy=False)})
    health = router.health()
    assert not health.healthy
    assert health.routes["default"][0].reason_code == "adapter_unhealthy"
    with pytest.raises(RouteUnavailable):
        router.complete(llm_request, caller=caller)
