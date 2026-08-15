# jainsons-llm-router

`jainsons-llm-router` is a small in-process Python library for safely making
production LLM calls. Applications create one `Router` from trusted route
policy, adapters, a durable ledger, and an optional structured logger. The
router chooses only configured headless free routes by default. It may use a
paid provider only when an expiring deployment approval and every relevant
daily cap allow it.

It is deliberately a library, not a CLI wrapper. It never starts an
interactive tool or provider CLI from a web request, worker, or cron job.

## What it owns

- Ordered, policy-defined free candidates and explicitly permitted paid fallbacks.
- Exact provider/model selections that never silently substitute another model.
- A same-host `FileLedger` that reserves integer micro-USD and call budgets
  before paid network dispatch.
- Conservative settlement: a timeout after dispatch remains charged until
  reconciliation proves the outcome.
- Structured privacy-safe events. Prompts, completions, credentials, raw
  idempotency keys, raw account aliases, and phone numbers are not logged.

GLM is structurally rejected from automatic route candidates. A configured key
does not change that behavior.

## Install

Until a private package registry is available, consume an immutable tag:

```bash
pip install "git+https://<git-host>/<org>/jainsons-llm-router.git@<signed-tag>"
```

For local development:

```bash
python -m pip install -e '.[test]'
pytest
```

## Minimal free-only example

```python
from jainsons_llm_router import (
    BillingClass, BudgetCap, Candidate, CallerContext, FakeAdapter, FileLedger,
    LLMRequest, RoutePolicy, RouterConfig, UseRoute, create_router,
)

free = Candidate(
    provider="approved-free-service",
    model="approved-free-model",
    adapter="free",
    billing_class=BillingClass.FREE,
    provider_account_alias="free-account",
    zero_marginal_cost=True,
)
route = RoutePolicy(name="customer_chat", free_candidates=(free,))
ledger = FileLedger.initialize(
    "/var/lib/my-service/llm-ledger",
    spend_domain="my-service-prod",
    # Paid scopes are still provisioned up front; paid use remains disabled.
    caps={"aggregate": BudgetCap(1, 1), "provider": BudgetCap(1, 1), "route": BudgetCap(1, 1)},
    price_card_versions={"price-card-v1"},
)
router = create_router(
    RouterConfig(policy_version="policy-v1", routes={"customer_chat": route}),
    adapters={"free": FakeAdapter(nonbillable_models={"approved-free-model"})},
    ledger=ledger,
)
result = router.complete(
    LLMRequest(input="Hello"),
    selection=UseRoute("customer_chat"),
    caller=CallerContext(
        service="my-service", environment="production", route_purpose="customer_chat",
        deployment_version="2026.08.15", correlation_id="privacy-safe-request-id",
    ),
)
```

The example uses `FakeAdapter` only to make the policy shape clear. Production
code should use `GenericHTTPAdapter` for an approved OpenAI-shaped HTTP service
or an adapter that implements the same `ProviderAdapter` contract.

## Paid routes and the ledger

A paid `Candidate` must name aggregate, provider, and route scopes, an account
alias, and a known price-card version. Each scope must have positive integer
call and micro-USD caps in the `FileLedger`. Use `Asia/Kolkata` unless the
account owner explicitly chooses a different account timezone.

Before dispatch, the router asks the adapter for a bounded maximum charge and
the ledger appends+fsyncs `RESERVE`. The ledger uses a dedicated `ledger.lock`
and stores `snapshot.json`, `journal.jsonl`, `archive/`, and `recovery/` under
the configured directory. It records `RESERVE`, `SETTLE`, `RELEASE`,
`SETTLE_UNKNOWN`, `OVERAGE`, `CHECKPOINT`, and `RECOVERY` events with a
tamper-evident sequence/digest chain.

The file ledger is safe for concurrent processes on one host with one shared
volume. It is not a cross-host authority. Multi-host paid deployments must use
one shared ledger service or remain free-only.

## Adapter boundary

Every adapter validates capability, estimates a bounded charge, sends sync and
async requests, returns normalized usage/request IDs, reports health, proves a
free model is non-billable, and marks failures as pre-dispatch, definitive, or
unknown. Adapters do not choose fallbacks or own a budget counter.

`NullAdapter` and `FakeAdapter` are included for disabled routes and tests.
`AnthropicAdapter`, `GeminiAdapter`, `OpenRouterAdapter`, and `MistralAdapter`
are safe interface-compatible stubs until an approved direct SDK integration is
configured.

## Maintainer checklist

- Treat route policy and paid approval records as deployment-owned, reviewed
  configuration; an API key alone never permits spending.
- Initialize the ledger in deployment provisioning, never lazily in a request.
- Keep every service that can charge the same provider account on the same
  spend-domain ledger authority.
- Do not add GLM to an automatic route.
- Keep price cards and cap scopes explicit and integer-only. Missing or invalid
  ledger data must fail closed for paid use while leaving free routes usable.
- Reconcile unknown reservations with provider evidence via a new `RECOVERY`
  event; never edit a journal line.
