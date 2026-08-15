from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from jainsons_llm_router import (
    BudgetCap,
    BudgetDenied,
    FileLedger,
    LedgerUnavailable,
    ReservationSpec,
    hash_private_identifier,
)


CAPS = {
    "agg": BudgetCap(5, 500),
    "provider": BudgetCap(5, 500),
    "route": BudgetCap(5, 500),
}


def reservation_spec(index: int, *, price_card_version: str = "prices-v1") -> ReservationSpec:
    return ReservationSpec(
        spend_domain="domain",
        aggregate_scope="agg",
        provider_scope="provider",
        route_scope="route",
        model_scope=None,
        provider_account_alias="account",
        provider="provider",
        model="model",
        service="service",
        environment="test",
        route="route",
        policy_version="policy-v1",
        caller_id="worker",
        correlation_id_hash=hash_private_identifier(f"correlation-{index}"),
        idempotency_key_hash=hash_private_identifier(f"idempotency-{index}"),
        reserved_micro_usd=100,
        price_card_version=price_card_version,
    )


def _reserve_in_process(directory: str, index: int, results) -> None:
    ledger = FileLedger(
        directory,
        spend_domain="domain",
        caps=CAPS,
        price_card_versions={"prices-v1"},
        allowed_provider_accounts={"account"},
    )
    try:
        ledger.reserve(reservation_spec(index))
    except BudgetDenied:
        results.put("denied")
    except Exception as exc:  # pragma: no cover - failures are asserted by parent
        results.put(f"error:{type(exc).__name__}")
    else:
        results.put("reserved")


def test_many_processes_cannot_exceed_atomic_call_or_money_cap(tmp_path):
    directory = tmp_path / "ledger"
    ledger = FileLedger.initialize(
        directory,
        spend_domain="domain",
        caps=CAPS,
        price_card_versions={"prices-v1"},
        allowed_provider_accounts={"account"},
    )
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    workers = [
        context.Process(target=_reserve_in_process, args=(str(directory), index, results))
        for index in range(16)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0
    outcomes = [results.get(timeout=5) for _ in workers]
    assert outcomes.count("reserved") == 5
    assert outcomes.count("denied") == 11
    assert not [result for result in outcomes if result.startswith("error:")]
    # Live reservations count towards every cap, before provider dispatch.
    assert ledger.remaining_budget("agg", caller=None).calls_remaining == 0
    assert ledger.remaining_budget("agg", caller=None).micro_usd_remaining == 0


def test_integrity_failures_and_unknown_price_card_fail_closed(tmp_path, monkeypatch):
    ledger = FileLedger.initialize(
        tmp_path / "ledger",
        spend_domain="domain",
        caps=CAPS,
        price_card_versions={"prices-v1"},
        allowed_provider_accounts={"account"},
    )
    with pytest.raises(LedgerUnavailable, match="unknown price-card"):
        ledger.reserve(reservation_spec(1, price_card_version="unknown"))

    ledger.lock_path.chmod(0)
    with pytest.raises(LedgerUnavailable, match="lock"):
        ledger.reserve(reservation_spec(2))
    ledger.lock_path.chmod(0o600)

    monkeypatch.setattr("jainsons_llm_router.ledger.os.fsync", lambda _fd: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises(LedgerUnavailable, match="append"):
        ledger.reserve(reservation_spec(3))


def test_compaction_preserves_state_and_keeps_archive_chain(tmp_path):
    ledger = FileLedger.initialize(
        tmp_path / "ledger",
        spend_domain="domain",
        caps=CAPS,
        price_card_versions={"prices-v1"},
        allowed_provider_accounts={"account"},
    )
    reservation = ledger.reserve(reservation_spec(1))
    ledger.settle(reservation.reservation_id, actual_micro_usd=80, input_tokens=2, output_tokens=3)
    before = ledger.remaining_budget("agg", caller=None)
    archive = ledger.compact()
    after = ledger.remaining_budget("agg", caller=None)
    assert archive.name.endswith(".jsonl.zst")
    assert archive.exists() and archive.stat().st_size > 0
    assert (before.calls_remaining, before.micro_usd_remaining) == (
        after.calls_remaining,
        after.micro_usd_remaining,
    )
    assert ledger.diagnose()["healthy"]
