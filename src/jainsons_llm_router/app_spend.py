"""Small, policy-independent accounting store for paid application lanes."""

from __future__ import annotations

import contextlib
from datetime import datetime
import json
import math
import os
import re
import tempfile
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import fcntl

from .app_alerts import emit_alert
from .errors import BudgetExhausted


_RESERVATION_TTL_SECONDS = 10 * 60
_BUDGET_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class AppSpend:
    """Atomically account for one lane's daily cap without price-card coupling."""

    def __init__(self, directory: str | os.PathLike[str], lane: str, cap_usd: float, budget_day_tz: str) -> None:
        self.directory = Path(directory)
        self.lane = lane
        self.cap_usd = float(cap_usd)
        self.timezone = ZoneInfo(budget_day_tz)

    def _day(self) -> str:
        return datetime.now(self.timezone).strftime("%Y-%m-%d")

    def _path(self, day: str | None = None) -> Path:
        return self.directory / f"{self.lane}-{day or self._day()}.json"

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"spent_usd": 0.0, "reservations": {}, "alerts_sent": []}

    @staticmethod
    def _normalise(raw: Any) -> dict[str, Any]:
        spent = raw.get("spent_usd")
        reservations = raw.get("reservations")
        alerts = raw.get("alerts_sent")
        clean_reservations: dict[str, dict[str, Any]] = {}
        for reservation_id, item in reservations.items():
            clean_item: dict[str, Any] = {"usd": float(item["usd"]), "ts": float(item["ts"])}
            if isinstance(item.get("day"), str):
                clean_item["day"] = item["day"]
            clean_reservations[reservation_id] = clean_item
        return {
            "spent_usd": float(spent),
            "reservations": clean_reservations,
            "alerts_sent": list(alerts),
        }

    @staticmethod
    def _is_valid(raw: Any) -> bool:
        if not isinstance(raw, dict) or set(raw) != {"spent_usd", "reservations", "alerts_sent"}:
            return False
        spent = raw["spent_usd"]
        if (
            not isinstance(spent, (int, float))
            or isinstance(spent, bool)
            or not math.isfinite(spent)
            or spent < 0
            or not isinstance(raw["reservations"], dict)
            or not isinstance(raw["alerts_sent"], list)
            or not all(isinstance(item, str) for item in raw["alerts_sent"])
        ):
            return False
        for reservation_id, item in raw["reservations"].items():
            if not isinstance(reservation_id, str) or not isinstance(item, dict):
                return False
            usd, stamp = item.get("usd"), item.get("ts")
            if (
                not isinstance(usd, (int, float))
                or isinstance(usd, bool)
                or not math.isfinite(usd)
                or usd < 0
                or not isinstance(stamp, (int, float))
                or isinstance(stamp, bool)
                or not math.isfinite(stamp)
            ):
                return False
            day = item.get("day")
            if day is not None and (not isinstance(day, str) or not _BUDGET_DAY.fullmatch(day)):
                return False
        return True

    @staticmethod
    def _expire(state: dict[str, Any], now: float) -> None:
        state["reservations"] = {
            key: value for key, value in state["reservations"].items()
            if now - value["ts"] <= _RESERVATION_TTL_SECONDS
        }

    @contextlib.contextmanager
    def _locked_state(self, day: str | None = None) -> Iterator[tuple[Path, dict[str, Any]]]:
        self.directory.mkdir(parents=True, exist_ok=True)
        budget_day = day or self._day()
        path = self._path(budget_day)
        lock_path = path.with_suffix(path.suffix + ".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not self._is_valid(raw):
                    raise ValueError("invalid spend state")
            except FileNotFoundError:
                raw = self._empty()
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                corrupt_path = path.with_name(f"{path.name}.corrupt-{time.time_ns()}")
                try:
                    os.replace(path, corrupt_path)
                except OSError:
                    pass
                raw = self._empty()
                raw["spent_usd"] = self.cap_usd
                self._write(path, raw)
                event = {
                    "type": "spend_corrupt",
                    "reason": "unreadable_or_corrupt",
                    "backend": "",
                    "policy_sha256": "",
                    "release": "",
                    "correlation_id": "",
                }
                try:
                    alert_directory = self.directory.parent if self.directory.name == "spend" else self.directory
                    emit_alert(
                        alert_directory,
                        event,
                        once_per_budget_day=True,
                        budget_day=budget_day,
                    )
                except Exception:
                    pass
            state = self._normalise(raw)
            self._expire(state, time.time())
            yield path, state
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @staticmethod
    def _write(path: Path, state: dict[str, Any]) -> None:
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass

    def reserve(self, estimated_usd: float) -> str:
        estimate = max(0.0, float(estimated_usd))
        day = self._day()
        with self._locked_state(day) as (path, state):
            open_total = sum(item["usd"] for item in state["reservations"].values())
            if state["spent_usd"] + open_total + estimate > self.cap_usd:
                self._write(path, state)  # persist expiry even when refusing.
                raise BudgetExhausted("application chat daily cap exhausted")
            reservation_id = f"{day}:{uuid.uuid4().hex}"
            state["reservations"][reservation_id] = {"usd": estimate, "ts": time.time(), "day": day}
            self._write(path, state)
            return reservation_id

    def _reservation_day(self, reservation_id: str) -> str:
        candidate, separator, _ = reservation_id.partition(":")
        if separator and _BUDGET_DAY.fullmatch(candidate):
            try:
                datetime.strptime(candidate, "%Y-%m-%d")
            except ValueError:
                pass
            else:
                return candidate
        return self._day()

    def settle(self, reservation_id: str, actual_usd: float) -> None:
        with self._locked_state(self._reservation_day(reservation_id)) as (path, state):
            if state["reservations"].pop(reservation_id, None) is not None:
                state["spent_usd"] += max(0.0, float(actual_usd))
            self._write(path, state)

    def settle_unknown(self, reservation_id: str) -> None:
        with self._locked_state(self._reservation_day(reservation_id)) as (path, state):
            reservation = state["reservations"].pop(reservation_id, None)
            if reservation is not None:
                state["spent_usd"] += reservation["usd"]
            self._write(path, state)

    def release(self, reservation_id: str) -> None:
        with self._locked_state(self._reservation_day(reservation_id)) as (path, state):
            state["reservations"].pop(reservation_id, None)
            self._write(path, state)

    def alert_once(self, alert_name: str) -> bool:
        with self._locked_state() as (path, state):
            if alert_name in state["alerts_sent"]:
                self._write(path, state)
                return False
            state["alerts_sent"].append(alert_name)
            self._write(path, state)
            return True

    def spent_today(self, *, read_only: bool = False) -> float:
        day = self._day()
        path = self._path(day)
        if read_only:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not self._is_valid(raw):
                    raise ValueError("invalid spend state")
                return self._normalise(raw)["spent_usd"]
            except FileNotFoundError:
                return 0.0
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                with self._locked_state(day) as (locked_path, state):
                    self._write(locked_path, state)
                    return state["spent_usd"]
        with self._locked_state(day) as (path, state):
            self._write(path, state)
            return state["spent_usd"]
