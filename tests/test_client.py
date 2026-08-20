from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from jainsons_llm_router import (
    BillingClass,
    ConfigurationError,
    LedgerUnavailable,
    client,
)
from jainsons_llm_router.policies.harness_derived import (
    BACKENDS,
    free_candidate_backend_order,
)


def test_lane_resolution_picks_first_concrete_free_model() -> None:
    candidates = client._lane_candidates("research")
    expected_backend = next(
        name
        for name in free_candidate_backend_order("research")
        if BACKENDS[name].automatic_enabled and BACKENDS[name].model
    )

    assert candidates[0].provider == expected_backend
    assert candidates[0].model == BACKENDS[expected_backend].model
    assert all(candidate.billing_class is BillingClass.FREE for candidate in candidates)
    assert all(candidate.zero_marginal_cost for candidate in candidates)


def test_free_only_ledger_refuses_settlement() -> None:
    with pytest.raises(LedgerUnavailable, match="free-only"):
        client._FreeOnlyLedger().settle("reservation", actual_micro_usd=0)


def test_unknown_lane_raises_clear_error_without_building_router(monkeypatch) -> None:
    def unexpected_router(*args, **kwargs):
        raise AssertionError("router should not be built for an unknown lane")

    monkeypatch.setattr(client, "_get_router", unexpected_router)
    with pytest.raises(ConfigurationError, match="unknown harness lane"):
        client.complete_text("hello", lane="does-not-exist")


def test_module_import_has_no_router_or_cli_side_effects() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    pythonpath = os.pathsep.join(
        part for part in (str(source_root), os.environ.get("PYTHONPATH", "")) if part
    )
    code = (
        "import jainsons_llm_router.client as client; "
        "assert client._ROUTERS == {}; "
        "print('import-clean')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )

    assert completed.stdout.strip() == "import-clean"
    assert completed.stderr == ""
