from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from jainsons_llm_router import (
    BillingClass,
    BudgetCap,
    CallerContext,
    Candidate,
    FileLedger,
    LLMRequest,
    PaidApproval,
)


@pytest.fixture
def caller() -> CallerContext:
    return CallerContext(
        service="svc",
        environment="test",
        route_purpose="default",
        deployment_version="v1",
        correlation_id="corr-1",
        caller_id="worker",
    )


@pytest.fixture
def llm_request() -> LLMRequest:
    return LLMRequest(input="hello", max_output_tokens=20, idempotency_key="stable-key")


def free_candidate(name: str, *, model: str | None = None) -> Candidate:
    return Candidate(
        provider=name,
        model=model or f"{name}-model",
        adapter=name,
        billing_class=BillingClass.FREE,
        provider_account_alias=f"{name}-free-account",
        zero_marginal_cost=True,
    )


def paid_candidate(
    name: str,
    *,
    model: str | None = None,
    aggregate_scope: str = "agg",
    provider_scope: str | None = None,
    route_scope: str = "route",
    price_card_version: str = "prices-v1",
) -> Candidate:
    return Candidate(
        provider=name,
        model=model or f"{name}-model",
        adapter=name,
        billing_class=BillingClass.PAID,
        provider_account_alias=f"{name}-paid-account",
        aggregate_scope=aggregate_scope,
        provider_scope=provider_scope or f"provider:{name}",
        route_scope=route_scope,
        price_card_version=price_card_version,
    )


def approval(*accounts: str, route: str = "default") -> PaidApproval:
    return PaidApproval(
        environment="test",
        route=route,
        service="svc",
        policy_version="policy-v1",
        allowed_provider_accounts=frozenset(accounts),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        approval_id="approval-1",
        enabled=True,
    )


def make_ledger(tmp_path, *, caps=None, prices=None, accounts=None) -> FileLedger:
    return FileLedger.initialize(
        tmp_path / "ledger",
        spend_domain="test-domain",
        caps=caps
        or {
            "agg": BudgetCap(20, 20_000),
            "route": BudgetCap(20, 20_000),
            "provider:paid": BudgetCap(20, 20_000),
            "provider:gemini": BudgetCap(20, 20_000),
            "provider:anthropic": BudgetCap(20, 20_000),
        },
        price_card_versions=frozenset(prices or {"prices-v1"}),
        allowed_provider_accounts=None if accounts is None else frozenset(accounts),
    )
