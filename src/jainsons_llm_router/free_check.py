"""Fail-closed live verification for OpenRouter's free-model catalog."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
from typing import Any
from urllib.request import Request, urlopen

PORTED_FROM_HARNESS_SHA256 = "012725adf46246795134fe835f9be6c4ebefd6304f3c276003265fd7cc37186b"  # harness openrouter_free_catalog._openrouter_model_is_free (V82)
_MODELS_URL = "https://openrouter.ai/api/v1/models"
def _fetch_models() -> tuple[dict[str, Any], ...] | None:
    """Fetch and validate a fresh catalog payload; catalog failures fail closed.

    The harness checks the catalog for every OpenRouter-free dispatch. Caching
    here would let a model remain eligible after OpenRouter changes its price.
    """
    try:
        request = Request(_MODELS_URL, headers={"Accept": "application/json"})
        with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed HTTPS API URL
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            return None
        return tuple(data)
    except Exception:  # Network and catalog errors must never make a model eligible.
        return None


def _is_exact_zero(value: Any) -> bool:
    """Accept only finite numeric values that are exactly zero (mirrors the harness)."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return False
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return False
    return number.is_finite() and number == 0


def _has_unambiguous_zero_io_pricing(pricing: Any) -> bool:
    """Every supplied input alias (prompt/input) and output alias (completion/output) must be zero."""
    if not isinstance(pricing, dict):
        return False
    input_keys = tuple(key for key in ("prompt", "input") if key in pricing)
    output_keys = tuple(key for key in ("completion", "output") if key in pricing)
    if not input_keys or not output_keys:
        return False
    return all(_is_exact_zero(pricing[key]) for key in (*input_keys, *output_keys))


def model_is_free(model_id: str) -> bool:
    """Whether OpenRouter currently declares exactly this ``:free`` model free.

    This ports the harness check: model IDs must have the suffix, catalog lookup
    must be unambiguous, and both prompt and completion pricing must be zero.
    """
    if not isinstance(model_id, str) or not model_id.endswith(":free"):
        return False
    models = _fetch_models()
    if models is None:
        return False
    matches = [model for model in models if model.get("id") == model_id]
    if len(matches) != 1:
        return False
    return _has_unambiguous_zero_io_pricing(matches[0].get("pricing"))
