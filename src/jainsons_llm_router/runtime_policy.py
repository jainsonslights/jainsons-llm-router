"""Validated, reloadable promoted app-routing policy.

This module deliberately consumes only the app policy in the harness export.
The existing free-worker policy remains isolated in ``sync_from_harness``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .app_alerts import emit_alert
from .errors import ConfigurationError, RouteUnavailable


DEFAULT_POLICY_PATH = Path("/opt/aria-brain/data/llm_router/policy.json")
POLICY_PATH_ENV = "JAINSONS_LLM_ROUTER_POLICY_PATH"
RELOAD_INTERVAL_SECONDS = 30.0
HOST_KEY_ENVS = {
    "api.deepseek.com": "DEEPSEEK_API_KEY",
    "openrouter.ai": "OPENROUTER_API_KEY",
}
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {"model", "messages", "max_tokens", "max_completion_tokens", "stream", "n", "tools", "tool_choice"}
)
_KEY_LIKE_VALUE = re.compile(
    r"(?:^sk-[A-Za-z0-9]|(?:api|access|auth|secret|private)[_-]?(?:key|token)|bearer\s+\S)", re.IGNORECASE
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_app_policy_sha256(app_backends: Mapping[str, Any], app_lanes: Mapping[str, Any]) -> str:
    """Return the exact canonical digest produced by the harness export."""

    payload = {"app_backends": app_backends, "app_lanes": app_lanes}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AppPolicy:
    path: Path
    app_backends: Mapping[str, Mapping[str, Any]]
    app_lanes: Mapping[str, Mapping[str, Any]]
    sha256: str
    release: str
    loaded_at: str
    mtime_ns: int
    file_size: int

    def lane(self, name: str) -> Mapping[str, Any]:
        try:
            return self.app_lanes[name]
        except KeyError as exc:
            raise RouteUnavailable("requested app lane is unavailable") from exc

    @property
    def lanes(self) -> Mapping[str, "AppLane"]:
        """Typed compatibility view for the forthcoming app-chat dispatcher."""

        return MappingProxyType({name: AppLane(name, lane) for name, lane in self.app_lanes.items()})


@dataclass(frozen=True)
class AppLane:
    name: str
    raw: Mapping[str, Any]

    @property
    def backend_chain(self) -> tuple[str, ...]:
        return tuple(self.raw["backend_chain"])


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"app policy {field} must be an object")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"app policy {field} must be a non-empty string")
    return value


_PAYLOAD_KEYS_BY_HOST = {
    "api.deepseek.com": frozenset({"thinking"}),
    "openrouter.ai": frozenset({"provider"}),
}
_PROVIDER_KEYS = frozenset({"data_collection", "order", "allow_fallbacks", "only", "ignore", "sort"})


def _validate_scalar_or_string_list(value: Any, path: str) -> None:
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ConfigurationError(f"app policy {path} must be a scalar or list of strings")
        for index, item in enumerate(value):
            _validate_payload_defaults(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        raise ConfigurationError(f"app policy {path} must be a scalar or list of strings")
    _validate_payload_defaults(value, path)


def _validate_payload_defaults(value: Any, path: str = "payload_defaults", *, host: str | None = None) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ConfigurationError(f"app policy {path} keys must be strings")
            if key in _FORBIDDEN_PAYLOAD_KEYS:
                raise ConfigurationError(f"app policy {path} contains protected request key")
            if path == "payload_defaults" and host is not None and key not in _PAYLOAD_KEYS_BY_HOST[host]:
                raise ConfigurationError(f"app policy {path} contains an unapproved key for its host")
            if path == "payload_defaults" and host == "api.deepseek.com" and key == "thinking" and not isinstance(child, dict):
                raise ConfigurationError("app policy deepseek thinking defaults must be an object")
            if path == "payload_defaults" and host == "openrouter.ai" and key == "provider" and not isinstance(child, dict):
                raise ConfigurationError("app policy openrouter provider defaults must be an object")
            if path == "payload_defaults.provider":
                if key not in _PROVIDER_KEYS:
                    raise ConfigurationError(f"app policy {path} contains an unapproved provider key")
                _validate_scalar_or_string_list(child, f"{path}.{key}")
            else:
                _validate_payload_defaults(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_payload_defaults(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if len(value) > 64 or _KEY_LIKE_VALUE.search(value):
            raise ConfigurationError(f"app policy {path} contains a key-like or oversized string")
    elif value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigurationError(f"app policy {path} contains a non-finite number")
    else:
        raise ConfigurationError(f"app policy {path} contains an unsupported value")


def _validate_backend(name: str, raw: Any) -> None:
    backend = _require_mapping(raw, f"app_backends[{name!r}]")
    model = _require_string(backend.get("model"), f"app_backends[{name!r}].model")
    if model.endswith(":free"):
        raise ConfigurationError("app policy may not route an OpenRouter free model")
    transport = _require_mapping(backend.get("transport"), f"app_backends[{name!r}].transport")
    if transport.get("dialect") != "openai-chat-completions":
        raise ConfigurationError("app policy transport dialect is not approved")
    endpoint = _require_string(transport.get("endpoint"), f"app_backends[{name!r}].transport.endpoint")
    allowed_host = _require_string(transport.get("allowed_host"), f"app_backends[{name!r}].transport.allowed_host")
    parsed = urlsplit(endpoint)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ConfigurationError("app policy endpoint has an invalid port") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != allowed_host
        or parsed.username
        or parsed.password
        or port not in (None, 443)
    ):
        raise ConfigurationError("app policy endpoint is not bound to its approved HTTPS host")
    expected_key_env = HOST_KEY_ENVS.get(allowed_host)
    if expected_key_env is None or transport.get("key_env") != expected_key_env:
        raise ConfigurationError("app policy host/key environment binding is not approved")
    defaults = _require_mapping(transport.get("payload_defaults", {}), f"app_backends[{name!r}].payload_defaults")
    _validate_payload_defaults(defaults, host=allowed_host)
    card = _require_mapping(backend.get("price_card"), f"app_backends[{name!r}].price_card")
    _require_string(card.get("version"), f"app_backends[{name!r}].price_card.version")
    for field, alternate in (
        ("input_usd_per_million", "input"),
        ("output_usd_per_million", "output"),
    ):
        value = card.get(field, card.get(alternate))
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ConfigurationError(f"app policy app_backends[{name!r}].price_card.{field} is invalid")
    minimum = backend.get("min_request_max_tokens")
    if minimum is not None and (
        not isinstance(minimum, int) or isinstance(minimum, bool) or not 1 <= minimum <= 16000
    ):
        raise ConfigurationError(f"app policy app_backends[{name!r}].min_request_max_tokens is invalid")


def _validate_lane(name: str, raw: Any, backends: Mapping[str, Any]) -> None:
    lane = _require_mapping(raw, f"app_lanes[{name!r}]")
    chain = lane.get("backend_chain")
    if not isinstance(chain, list) or not chain or not all(isinstance(item, str) and item for item in chain):
        raise ConfigurationError("app policy backend_chain must be a non-empty string list")
    if any(item not in backends for item in chain):
        raise ConfigurationError("app policy lane references an undeclared backend")

    def bounded_number(field: str, low: float, high: float, *, strict_low: bool = False) -> None:
        value = lane.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or (value <= low if strict_low else value < low)
            or value > high
        ):
            raise ConfigurationError(f"app policy app_lanes[{name!r}].{field} is invalid")

    bounded_number("daily_cap_usd", 0, 50, strict_low=True)
    for field, high in (("max_output_tokens", 8000), ("max_input_chars", 500000), ("max_deadline_seconds", 120)):
        value = lane.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > high:
            raise ConfigurationError(f"app policy app_lanes[{name!r}].{field} is invalid")
    bounded_number("primary_share", 0, 1, strict_low=True)
    fractions = lane.get("cap_alert_fractions")
    if (
        not isinstance(fractions, list)
        or not fractions
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            or value > 1
            for value in fractions
        )
    ):
        raise ConfigurationError(f"app policy app_lanes[{name!r}].cap_alert_fractions is invalid")
    timezone_name = lane.get("budget_day_tz")
    if not isinstance(timezone_name, str) or not timezone_name:
        raise ConfigurationError(f"app policy app_lanes[{name!r}].budget_day_tz is invalid")
    try:
        ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ConfigurationError(f"app policy app_lanes[{name!r}].budget_day_tz is invalid") from None


def validate_app_policy(
    raw: Any,
    path: str | os.PathLike[str],
    *,
    file_stat: os.stat_result | None = None,
) -> AppPolicy:
    """Validate one JSON export and return an immutable app-policy view."""

    root = _require_mapping(raw, "root")
    if root.get("schema") != 1:
        raise ConfigurationError("app policy schema must be 1")
    backends = _require_mapping(root.get("app_backends"), "app_backends")
    lanes = _require_mapping(root.get("app_lanes"), "app_lanes")
    if not backends or not lanes:
        raise ConfigurationError("app policy must declare app backends and lanes")
    declared_sha = _require_string(root.get("app_policy_sha256"), "app_policy_sha256")
    if not _SHA256.fullmatch(declared_sha):
        raise ConfigurationError("app policy sha must be a sha256 hex digest")
    if declared_sha != canonical_app_policy_sha256(backends, lanes):
        raise ConfigurationError("app policy sha does not match canonical app policy")
    release = _require_string(root.get("release"), "release")
    for name, backend in backends.items():
        if not isinstance(name, str) or not name:
            raise ConfigurationError("app policy backend names must be non-empty strings")
        _validate_backend(name, backend)
    for lane_name, lane_raw in lanes.items():
        if not isinstance(lane_name, str) or not lane_name:
            raise ConfigurationError("app policy lane names must be non-empty strings")
        _validate_lane(lane_name, lane_raw, backends)
    if file_stat is None:
        try:
            with Path(path).open("rb") as handle:
                file_stat = os.fstat(handle.fileno())
        except OSError as exc:
            raise ConfigurationError("app policy cannot be statted") from exc
    return AppPolicy(
        path=Path(path),
        app_backends=MappingProxyType({name: MappingProxyType(dict(value)) for name, value in backends.items()}),
        app_lanes=MappingProxyType({name: MappingProxyType(dict(value)) for name, value in lanes.items()}),
        sha256=declared_sha,
        release=release,
        loaded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        mtime_ns=file_stat.st_mtime_ns,
        file_size=file_stat.st_size,
    )


class RuntimePolicy:
    """Mtime-aware policy holder that safely retains its previous good policy."""

    def __init__(
        self,
        policy_path: str | os.PathLike[str] | None = None,
        *,
        reload_interval_seconds: float = RELOAD_INTERVAL_SECONDS,
        reload_interval: float | None = None,
    ) -> None:
        self.path = Path(policy_path or os.environ.get(POLICY_PATH_ENV, DEFAULT_POLICY_PATH))
        self.reload_interval_seconds = reload_interval_seconds if reload_interval is None else reload_interval
        self._last_check = float("-inf")
        self._policy: AppPolicy | None = None
        self._last_load_failed = False
        self._lock = threading.Lock()

    @property
    def last_load_failed(self) -> bool:
        with self._lock:
            return self._last_load_failed

    def get(self, *, force: bool = False) -> AppPolicy:
        with self._lock:
            now = time.monotonic()
            if not force and self._policy is not None and now - self._last_check < self.reload_interval_seconds:
                return self._policy
            self._last_check = now
            try:
                with self.path.open("rb") as handle:
                    stat = os.fstat(handle.fileno())
                    if (
                        self._policy is not None
                        and stat.st_mtime_ns == self._policy.mtime_ns
                        and stat.st_size == self._policy.file_size
                    ):
                        return self._policy
                    raw = json.load(handle)
                self._policy = validate_app_policy(raw, self.path, file_stat=stat)
                self._last_load_failed = False
                return self._policy
            except Exception:
                self._last_load_failed = True
                event = {
                    "type": "policy_invalid",
                    "reason": "invalid_or_unavailable",
                    "backend": "",
                    "policy_sha256": self._policy.sha256 if self._policy else "",
                    "release": self._policy.release if self._policy else "",
                    "correlation_id": "",
                }
                try:
                    emit_alert(self.path.parent, event)
                except Exception:
                    pass
                if self._policy is not None:
                    return self._policy
                raise RouteUnavailable("no valid promoted app policy is available") from None

    def load(self) -> AppPolicy:
        """Compatibility spelling for callers that use a loader-style API."""

        return self.get()

    def read_only(self) -> AppPolicy:
        """Load without mutating runtime state or emitting a durable alert."""

        try:
            with self.path.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                raw = json.load(handle)
            return validate_app_policy(raw, self.path, file_stat=stat)
        except Exception:
            raise RouteUnavailable("no valid promoted app policy is available") from None


_RUNTIMES: dict[Path, RuntimePolicy] = {}
_RUNTIMES_LOCK = threading.Lock()


def get_runtime_policy(policy_path: str | os.PathLike[str] | None = None) -> RuntimePolicy:
    path = Path(policy_path or os.environ.get(POLICY_PATH_ENV, DEFAULT_POLICY_PATH))
    with _RUNTIMES_LOCK:
        return _RUNTIMES.setdefault(path, RuntimePolicy(path))


def load_app_policy(policy_path: str | os.PathLike[str] | None = None) -> AppPolicy:
    return get_runtime_policy(policy_path).get()
