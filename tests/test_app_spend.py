from __future__ import annotations

import json
import multiprocessing
import time

import pytest

from jainsons_llm_router.app_spend import AppSpend
from jainsons_llm_router.errors import BudgetExhausted


def _reserve_once(directory: str, queue) -> None:
    try:
        AppSpend(directory, "chat_fast", 1.0, "Asia/Kolkata").reserve(0.6)
        queue.put(True)
    except BudgetExhausted:
        queue.put(False)


def test_over_cap_blocks_until_next_budget_day(tmp_path, monkeypatch):
    monkeypatch.setattr(AppSpend, "_day", lambda self: "2026-09-14")
    spend = AppSpend(tmp_path, "chat_fast", 1.0, "Asia/Kolkata")
    reservation = spend.reserve(0.1)
    spend.settle(reservation, 1.5)
    with pytest.raises(BudgetExhausted):
        spend.reserve(0.01)
    assert spend.spent_today(read_only=True) == 1.5
    monkeypatch.setattr(AppSpend, "_day", lambda self: "2026-09-15")
    assert spend.reserve(0.01)


def test_cap_lowered_below_existing_spend_blocks(tmp_path):
    original = AppSpend(tmp_path, "chat_fast", 1.0, "Asia/Kolkata")
    reservation = original.reserve(0.8)
    original.settle(reservation, 0.8)
    lowered = AppSpend(tmp_path, "chat_fast", 0.5, "Asia/Kolkata")
    with pytest.raises(BudgetExhausted):
        lowered.reserve(0.01)


def test_new_budget_day_and_expired_reservations_do_not_carry_forward(tmp_path, monkeypatch):
    monkeypatch.setattr(AppSpend, "_day", lambda self: "2026-09-14")
    spend = AppSpend(tmp_path, "chat_fast", 1.0, "Asia/Kolkata")
    reservation = spend.reserve(0.9)
    state_path = tmp_path / "chat_fast-2026-09-14.json"
    state = json.loads(state_path.read_text())
    state["reservations"][reservation]["ts"] = time.time() - 601
    state_path.write_text(json.dumps(state))
    assert spend.reserve(0.9)
    monkeypatch.setattr(AppSpend, "_day", lambda self: "2026-09-15")
    assert spend.spent_today(read_only=True) == 0.0


def test_concurrent_process_reservations_respect_cap(tmp_path):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_reserve_once, args=(str(tmp_path), queue)) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    assert sorted(queue.get(timeout=1) for _ in processes) == [False, True]


def test_price_changes_are_irrelevant_to_spend_files(tmp_path):
    spend = AppSpend(tmp_path, "chat_fast", 1.0, "Asia/Kolkata")
    reservation = spend.reserve(0.2)
    spend.settle(reservation, 0.2)
    state = json.loads(next(tmp_path.glob("chat_fast-*.json")).read_text())
    assert set(state) == {"spent_usd", "reservations", "alerts_sent"}


@pytest.mark.parametrize(
    "operation, expected_spend",
    [("settle", 0.2), ("settle_unknown", 0.2), ("release", 0.0)],
)
def test_reservation_updates_use_the_original_budget_day(tmp_path, monkeypatch, operation, expected_spend):
    monkeypatch.setattr(AppSpend, "_day", lambda self: "2026-09-14")
    spend = AppSpend(tmp_path, "chat_fast", 1.0, "Asia/Kolkata")
    reservation = spend.reserve(0.2)
    state = json.loads((tmp_path / "chat_fast-2026-09-14.json").read_text())
    assert state["reservations"][reservation]["day"] == "2026-09-14"

    monkeypatch.setattr(AppSpend, "_day", lambda self: "2026-09-15")
    if operation == "settle":
        spend.settle(reservation, 0.2)
    else:
        getattr(spend, operation)(reservation)
    old_state = json.loads((tmp_path / "chat_fast-2026-09-14.json").read_text())
    assert old_state["spent_usd"] == expected_spend
    assert old_state["reservations"] == {}
    assert spend.spent_today(read_only=True) == 0.0


def test_corrupt_spend_file_is_quarantined_alerted_and_blocks_for_day(tmp_path, monkeypatch):
    from jainsons_llm_router import set_alert_sink

    monkeypatch.setattr(AppSpend, "_day", lambda self: "2026-09-14")
    spend_dir = tmp_path / "spend"
    spend_dir.mkdir()
    path = spend_dir / "chat_fast-2026-09-14.json"
    path.write_text("{not-json", encoding="utf-8")
    seen = []
    set_alert_sink(seen.append)
    try:
        with pytest.raises(BudgetExhausted):
            AppSpend(spend_dir, "chat_fast", 1.0, "Asia/Kolkata").reserve(0.01)
    finally:
        set_alert_sink(None)
    assert len(list(spend_dir.glob("chat_fast-2026-09-14.json.corrupt-*"))) == 1
    assert json.loads(path.read_text())["spent_usd"] == 1.0
    assert [event["type"] for event in seen] == ["spend_corrupt"]

    monkeypatch.setattr(AppSpend, "_day", lambda self: "2026-09-15")
    assert AppSpend(spend_dir, "chat_fast", 1.0, "Asia/Kolkata").reserve(0.01)
