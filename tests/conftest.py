from __future__ import annotations

import os
import socket
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


@pytest.fixture(autouse=True)
def isolate_tests_from_credentials_and_network(tmp_path, monkeypatch):
    """Keep tests from reading real credentials or reaching non-loopback hosts."""

    for name in tuple(os.environ):
        if name in {"DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "LLM_DEEPSEEK_KEY"} or name.startswith(
            "JAINSONS_LLM_ROUTER_"
        ):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    real_create_connection = socket.create_connection
    real_socket_connect = socket.socket.connect
    loopback_hosts = {"127.0.0.1", "::1", "localhost"}

    def guarded_create_connection(address, *args, **kwargs):
        host = address[0]
        if host not in loopback_hosts:
            raise RuntimeError("network blocked in tests")
        return real_create_connection(address, *args, **kwargs)

    def guarded_socket_connect(sock, address):
        # Unix-domain socket addresses are filesystem paths rather than hosts.
        if isinstance(address, tuple) and address[0] not in loopback_hosts:
            raise RuntimeError("network blocked in tests")
        return real_socket_connect(sock, address)

    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket.socket, "connect", guarded_socket_connect)


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
