"""Fail-closed live verification for OpenRouter's free-model catalog."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
from typing import Any
from urllib.request import Request, urlopen

PORTED_FROM_HARNESS_SHA256 = "f7acf1713e37a0139eff17c9f04b51036c8af2400cbfb315bc02960f4e1e7b73"  # harness openrouter_free_catalog._openrouter_model_is_free (V121: strict catalog JSON)
_MODELS_URL = "https://openrouter.ai/api/v1/models"
_MAX_CATALOG_BYTES = 8 * 1024 * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {name}")


def _loads_strict(data: bytes) -> Any:
    """Mirror harness V121 loads_provider: bounded, duplicate-free, finite JSON."""
    if len(data) > _MAX_CATALOG_BYTES:
        raise ValueError("catalog JSON is too large")
    return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def _fetch_models() -> tuple[dict[str, Any], ...] | None:
    """Fetch and validate a fresh catalog payload; catalog failures fail closed.

    The harness checks the catalog for every OpenRouter-free dispatch. Caching
    here would let a model remain eligible after OpenRouter changes its price.
    """
    try:
        request = Request(_MODELS_URL, headers={"Accept": "application/json"})
        with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed HTTPS API URL
            payload = _loads_strict(response.read())
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return None
        # Mirrors the harness: unrelated malformed entries are ignored, never matched.
        return tuple(item for item in data if isinstance(item, dict))
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
    """Whether OpenRouter currently declares exactly this model free.

    This ports the harness check (V101): a ``:free`` suffix is not pricing
    evidence (stealth models can be free without it), so only the live catalog
    decides. An ``openrouter/`` prefix is dropped as in the harness, catalog
    lookup must be unambiguous, and both prompt and completion pricing must be
    exactly zero.
    """
    if not isinstance(model_id, str):
        return False
    requested_id = model_id.removeprefix("openrouter/")
    if not requested_id or requested_id == ":free":
        return False
    models = _fetch_models()
    if models is None:
        return False
    matches = [model for model in models if model.get("id") == requested_id]
    if len(matches) != 1:
        return False
    return _has_unambiguous_zero_io_pricing(matches[0].get("pricing"))
