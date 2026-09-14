"""Redacted, durable alerts for the promoted application-policy path."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

try:  # pragma: no cover - Windows fallback is platform-specific.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]
    import msvcrt  # type: ignore[import-not-found]


AlertSink = Callable[[dict[str, str]], None]
_FIELDS = ("type", "reason", "backend", "policy_sha256", "release", "correlation_id")
_SINK: AlertSink | None = None
_SINK_LOCK = threading.Lock()


def set_alert_sink(sink: AlertSink | None) -> None:
    """Set a process-local sink; ``None`` uses the durable JSONL default."""

    if sink is not None and not callable(sink):
        raise TypeError("alert sink must be callable or None")
    with _SINK_LOCK:
        global _SINK
        _SINK = sink


@contextlib.contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:  # pragma: no cover
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            else:  # pragma: no cover
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(fd)


def _read(path: Path) -> dict[str, float]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {key: float(value) for key, value in raw.items() if isinstance(key, str) and isinstance(value, (int, float))} if isinstance(raw, dict) else {}


def _write(path: Path, state: Mapping[str, float]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(state, handle, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def emit_alert(
    policy_dir: str | os.PathLike[str],
    event: Mapping[str, Any],
    *,
    once_per_budget_day: bool = False,
    budget_day: str | None = None,
) -> bool:
    """Append a safe event once per 5 minutes (or once per budget day)."""

    safe = {field: str(event.get(field, "")) for field in _FIELDS}
    directory = Path(policy_dir)
    ledger = directory / "ledger"
    key = f"{safe['type']}\0{safe['reason']}"
    if once_per_budget_day:
        key = f"{budget_day or ''}\0{key}"
    state_path = ledger / ("alert_budget_days.json" if once_per_budget_day else "alert_dedupe.json")
    now = time.time()
    with _lock(ledger / "alerts.lock"):
        state = _read(state_path)
        if once_per_budget_day:
            if key in state:
                return False
        else:
            state = {item: stamp for item, stamp in state.items() if now - stamp < 300}
            if key in state:
                _write(state_path, state)
                return False
        state[key] = now
        _write(state_path, state)
        with open(directory / "alerts.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    with _SINK_LOCK:
        sink = _SINK
    if sink is not None:
        try:
            sink(dict(safe))
        except Exception:
            pass
    return True
