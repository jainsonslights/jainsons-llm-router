"""Durable same-host usage ledger with atomic multi-scope reservations."""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import hashlib
import json
import os
import stat
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .errors import BudgetDenied, ConfigurationError, LedgerUnavailable
from .models import BudgetCap, BudgetRemaining, CallerContext, validate_identifier

try:  # pragma: no cover - Windows fallback is exercised only on Windows.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]
    import msvcrt  # type: ignore[import-not-found]


SCHEMA_VERSION = 1
GENESIS_DIGEST = "0" * 64
EVENT_TYPES = frozenset(
    {"RESERVE", "SETTLE", "RELEASE", "SETTLE_UNKNOWN", "OVERAGE", "CHECKPOINT", "RECOVERY"}
)
COMMON_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "sequence",
        "previous_digest",
        "digest",
        "occurred_at_utc",
        "budget_day",
        "timezone",
        "event_type",
        "spend_domain",
        "aggregate_scope",
        "provider_scope",
        "route_scope",
        "model_scope",
        "provider_account_alias",
        "provider",
        "model",
        "service",
        "environment",
        "route",
        "policy_version",
        "caller_id",
        "correlation_id",
        "idempotency_key_hash",
        "reservation_id",
        "provider_request_id_hash",
        "reserved_calls",
        "reserved_micro_usd",
        "actual_calls",
        "actual_micro_usd",
        "input_tokens",
        "output_tokens",
        "outcome",
        "reason",
        "attempt_number",
        "price_card_version",
    }
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest_mapping(value: Mapping[str, Any]) -> str:
    copy = dict(value)
    copy.pop("digest", None)
    return hashlib.sha256(_canonical_json(copy)).hexdigest()


def hash_private_identifier(value: str | None) -> str:
    return hashlib.sha256((value or "").encode("utf-8", "replace")).hexdigest()


@dataclass(frozen=True)
class ReservationSpec:
    spend_domain: str
    aggregate_scope: str
    provider_scope: str
    route_scope: str
    model_scope: str | None
    provider_account_alias: str
    provider: str
    model: str
    service: str
    environment: str
    route: str
    policy_version: str
    caller_id: str
    correlation_id_hash: str
    idempotency_key_hash: str
    reserved_micro_usd: int
    price_card_version: str
    attempt_number: int = 1

    def __post_init__(self) -> None:
        identifiers = (
            ("spend_domain", self.spend_domain),
            ("aggregate_scope", self.aggregate_scope),
            ("provider_scope", self.provider_scope),
            ("route_scope", self.route_scope),
            ("provider_account_alias", self.provider_account_alias),
            ("provider", self.provider),
            ("model", self.model),
            ("service", self.service),
            ("environment", self.environment),
            ("route", self.route),
            ("policy_version", self.policy_version),
            ("caller_id", self.caller_id),
            ("price_card_version", self.price_card_version),
        )
        for label, value in identifiers:
            validate_identifier(value, label)
        if self.model_scope:
            validate_identifier(self.model_scope, "model_scope")
        if not isinstance(self.reserved_micro_usd, int) or isinstance(self.reserved_micro_usd, bool):
            raise ConfigurationError("reservation money must be integer micro-USD")
        if self.reserved_micro_usd <= 0:
            raise ConfigurationError("paid reservation must be positive")
        if self.attempt_number <= 0:
            raise ConfigurationError("attempt_number must be positive")
        for label, value in (
            ("correlation_id_hash", self.correlation_id_hash),
            ("idempotency_key_hash", self.idempotency_key_hash),
        ):
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ConfigurationError(f"{label} must be a sha256 hex digest")

    @property
    def scopes(self) -> tuple[str, ...]:
        scopes = [self.aggregate_scope, self.provider_scope, self.route_scope]
        if self.model_scope:
            scopes.append(self.model_scope)
        return tuple(scopes)


@dataclass(frozen=True)
class LedgerReservation:
    reservation_id: str
    budget_day: str
    reserved_calls: int
    reserved_micro_usd: int
    scopes: tuple[str, ...]
    idempotency_key_hash: str


class IdempotencyConflict(BudgetDenied):
    code = "idempotency_conflict"

    def __init__(self, status: str) -> None:
        super().__init__("idempotency key already has a paid outcome", details={"status": status})
        self.status = status


class UsageLedger(Protocol):
    spend_domain: str

    def reserve(self, spec: ReservationSpec) -> LedgerReservation: ...

    def settle(
        self,
        reservation_id: str,
        *,
        actual_micro_usd: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        provider_request_id_hash: str | None = None,
        outcome: str = "success",
    ) -> None: ...

    def release(self, reservation_id: str, *, reason: str) -> None: ...

    def settle_unknown(
        self,
        reservation_id: str,
        *,
        reason: str,
        provider_request_id_hash: str | None = None,
    ) -> None: ...

    def remaining_budget(self, scope: str, *, caller: CallerContext) -> BudgetRemaining: ...


_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


def _process_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(key, threading.RLock())


def _new_state(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "totals": json.loads(json.dumps(snapshot.get("totals", {}))),
        "live_reservations": json.loads(json.dumps(snapshot.get("live_reservations", {}))),
        "idempotency": json.loads(json.dumps(snapshot.get("idempotency", {}))),
        "overage_blocked": bool(snapshot.get("overage_blocked", False)),
        "last_sequence": int(snapshot.get("checkpoint_sequence", 0)),
        "last_digest": str(snapshot.get("checkpoint_digest", GENESIS_DIGEST)),
    }


class FileLedger:
    """Same-host, file-locked JSONL authority for paid dispatch.

    Use :meth:`initialize` as an explicit deployment step. Normal construction
    never creates missing ledger files and therefore fails closed.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        spend_domain: str,
        caps: Mapping[str, BudgetCap],
        price_card_versions: set[str] | frozenset[str],
        timezone_name: str = "Asia/Kolkata",
        allowed_provider_accounts: set[str] | frozenset[str] | None = None,
        create: bool = False,
    ) -> None:
        validate_identifier(spend_domain, "spend_domain")
        self.directory = Path(directory)
        self.spend_domain = spend_domain
        self.caps = dict(caps)
        if not self.caps:
            raise ConfigurationError("paid ledger requires cap scopes")
        for scope, cap in self.caps.items():
            validate_identifier(scope, "cap scope")
            if not isinstance(cap, BudgetCap):
                raise ConfigurationError("each ledger cap must be a BudgetCap")
        self.price_card_versions = frozenset(price_card_versions)
        if not self.price_card_versions:
            raise ConfigurationError("paid ledger requires at least one price-card version")
        for version in self.price_card_versions:
            validate_identifier(version, "price-card version")
        try:
            self.timezone = ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ConfigurationError("invalid ledger timezone") from exc
        self.timezone_name = timezone_name
        self.allowed_provider_accounts = (
            None if allowed_provider_accounts is None else frozenset(allowed_provider_accounts)
        )
        if self.allowed_provider_accounts is not None:
            for alias in self.allowed_provider_accounts:
                validate_identifier(alias, "provider account alias")
        self.lock_path = self.directory / "ledger.lock"
        self.snapshot_path = self.directory / "snapshot.json"
        self.journal_path = self.directory / "journal.jsonl"
        self.archive_dir = self.directory / "archive"
        self.recovery_dir = self.directory / "recovery"
        self._thread_lock = _process_lock(self.lock_path)
        if create:
            self._initialize_files()

    @classmethod
    def initialize(
        cls,
        directory: str | os.PathLike[str],
        *,
        spend_domain: str,
        caps: Mapping[str, BudgetCap],
        price_card_versions: set[str] | frozenset[str],
        timezone_name: str = "Asia/Kolkata",
        allowed_provider_accounts: set[str] | frozenset[str] | None = None,
    ) -> "FileLedger":
        return cls(
            directory,
            spend_domain=spend_domain,
            caps=caps,
            price_card_versions=price_card_versions,
            timezone_name=timezone_name,
            allowed_provider_accounts=allowed_provider_accounts,
            create=True,
        )

    def _initialize_files(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        self.recovery_dir.mkdir(exist_ok=True)
        if any(path.exists() for path in (self.lock_path, self.snapshot_path, self.journal_path)):
            raise ConfigurationError("ledger already exists; initialization refuses to overwrite")
        lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(lock_fd)
        journal_fd = os.open(self.journal_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.fsync(journal_fd)
        finally:
            os.close(journal_fd)
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "spend_domain": self.spend_domain,
            "timezone": self.timezone_name,
            "checkpoint_sequence": 0,
            "checkpoint_digest": GENESIS_DIGEST,
            "totals": {},
            "live_reservations": {},
            "idempotency": {},
            "overage_blocked": False,
            "price_card_versions": sorted(self.price_card_versions),
        }
        self._atomic_write_snapshot(snapshot)
        self._fsync_directory(self.directory)

    def _budget_day(self, now: datetime | None = None) -> str:
        moment = now or datetime.now(timezone.utc)
        return moment.astimezone(self.timezone).date().isoformat()

    @contextlib.contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._thread_lock:
            try:
                mode = self.lock_path.stat().st_mode
                readable = mode & (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                writable = mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                if not readable or not writable:
                    raise PermissionError("ledger.lock lacks read/write permission")
                fd = os.open(self.lock_path, os.O_RDWR)
            except Exception as exc:
                raise LedgerUnavailable("cannot open ledger lock") from exc
            try:
                try:
                    if fcntl is not None:
                        fcntl.flock(fd, fcntl.LOCK_EX)
                    else:  # pragma: no cover
                        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                except Exception as exc:
                    raise LedgerUnavailable("cannot acquire ledger lock") from exc
                yield
            finally:
                try:
                    if fcntl is not None:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    else:  # pragma: no cover
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
                os.close(fd)

    def _read_snapshot(self) -> dict[str, Any]:
        try:
            raw = self.snapshot_path.read_text(encoding="utf-8")
            snapshot = json.loads(raw)
        except Exception as exc:
            raise LedgerUnavailable("snapshot cannot be read or parsed") from exc
        required = {
            "schema_version",
            "spend_domain",
            "timezone",
            "checkpoint_sequence",
            "checkpoint_digest",
            "totals",
            "live_reservations",
            "idempotency",
            "overage_blocked",
            "price_card_versions",
            "digest",
        }
        if not isinstance(snapshot, dict) or not required.issubset(snapshot):
            raise LedgerUnavailable("snapshot schema is incomplete")
        if snapshot["digest"] != _digest_mapping(snapshot):
            raise LedgerUnavailable("snapshot digest mismatch")
        if snapshot["schema_version"] != SCHEMA_VERSION:
            raise LedgerUnavailable("unknown snapshot schema version")
        if snapshot["spend_domain"] != self.spend_domain:
            raise LedgerUnavailable("unexpected spend domain")
        if snapshot["timezone"] != self.timezone_name:
            raise LedgerUnavailable("unexpected ledger timezone")
        if set(snapshot["price_card_versions"]) != set(self.price_card_versions):
            raise LedgerUnavailable("price-card configuration mismatch")
        return snapshot

    def _read_journal_events(self) -> list[dict[str, Any]]:
        try:
            raw = self.journal_path.read_bytes()
        except Exception as exc:
            raise LedgerUnavailable("journal cannot be read") from exc
        if raw and not raw.endswith(b"\n"):
            raise LedgerUnavailable("partial journal line")
        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line:
                raise LedgerUnavailable("blank journal line")
            try:
                event = json.loads(line)
            except Exception as exc:
                raise LedgerUnavailable(f"corrupt journal line {line_number}") from exc
            if not isinstance(event, dict):
                raise LedgerUnavailable("journal event is not an object")
            events.append(event)
        return events

    def _load_state(self) -> tuple[dict[str, Any], dict[str, Any]]:
        snapshot = self._read_snapshot()
        state = _new_state(snapshot)
        expected_sequence = state["last_sequence"] + 1
        previous_digest = state["last_digest"]
        for event in self._read_journal_events():
            if not COMMON_EVENT_FIELDS.issubset(event):
                raise LedgerUnavailable("journal event schema is incomplete")
            if event["schema_version"] != SCHEMA_VERSION or event["event_type"] not in EVENT_TYPES:
                raise LedgerUnavailable("unknown journal event version or type")
            if event["sequence"] != expected_sequence or event["previous_digest"] != previous_digest:
                raise LedgerUnavailable("journal sequence or digest chain is broken")
            if event["digest"] != _digest_mapping(event):
                raise LedgerUnavailable("journal event digest mismatch")
            if event["spend_domain"] != self.spend_domain or event["timezone"] != self.timezone_name:
                raise LedgerUnavailable("journal domain or timezone mismatch")
            self._apply_event(state, event)
            expected_sequence += 1
            previous_digest = event["digest"]
            state["last_sequence"] = event["sequence"]
            state["last_digest"] = event["digest"]
        return snapshot, state

    @staticmethod
    def _event_scopes(event: Mapping[str, Any]) -> tuple[str, ...]:
        values = [event["aggregate_scope"], event["provider_scope"], event["route_scope"]]
        if event.get("model_scope"):
            values.append(event["model_scope"])
        return tuple(value for value in values if value)

    @staticmethod
    def _add_totals(state: dict[str, Any], event: Mapping[str, Any]) -> None:
        day = event["budget_day"]
        day_totals = state["totals"].setdefault(day, {})
        for scope in FileLedger._event_scopes(event):
            current = day_totals.setdefault(scope, {"calls": 0, "micro_usd": 0})
            current["calls"] += int(event["actual_calls"])
            current["micro_usd"] += int(event["actual_micro_usd"])

    def _apply_event(self, state: dict[str, Any], event: Mapping[str, Any]) -> None:
        event_type = event["event_type"]
        reservation_id = event["reservation_id"]
        key_hash = event["idempotency_key_hash"]
        if event_type == "RESERVE":
            if reservation_id in state["live_reservations"] or key_hash in state["idempotency"]:
                raise LedgerUnavailable("duplicate reservation in journal")
            record = dict(event)
            state["live_reservations"][reservation_id] = record
            state["idempotency"][key_hash] = {"status": "live", "reservation_id": reservation_id}
        elif event_type == "RELEASE":
            record = state["live_reservations"].pop(reservation_id, None)
            if record is None:
                raise LedgerUnavailable("release references unknown reservation")
            state["idempotency"].pop(record["idempotency_key_hash"], None)
        elif event_type in {"SETTLE", "SETTLE_UNKNOWN"}:
            record = state["live_reservations"].pop(reservation_id, None)
            if record is None:
                raise LedgerUnavailable("settlement references unknown reservation")
            self._add_totals(state, event)
            if (
                event_type == "SETTLE"
                and event["outcome"] == "provider_failure"
                and int(event["actual_micro_usd"]) == 0
            ):
                # A definitive zero-charge provider rejection is resolved and
                # may fall through to the next explicitly listed paid sibling.
                state["idempotency"].pop(record["idempotency_key_hash"], None)
            else:
                status = "unknown" if event_type == "SETTLE_UNKNOWN" else "settled"
                state["idempotency"][record["idempotency_key_hash"]] = {
                    "status": status,
                    "reservation_id": reservation_id,
                }
        elif event_type == "OVERAGE":
            state["overage_blocked"] = True
        elif event_type == "RECOVERY":
            record = state["live_reservations"].pop(reservation_id, None)
            if record is None:
                raise LedgerUnavailable("recovery references unknown reservation")
            if event["outcome"] == "released":
                state["idempotency"].pop(record["idempotency_key_hash"], None)
            elif event["outcome"] == "settled":
                self._add_totals(state, event)
                state["idempotency"][record["idempotency_key_hash"]] = {
                    "status": "recovered",
                    "reservation_id": reservation_id,
                }
            else:
                raise LedgerUnavailable("unknown recovery outcome")

    def _base_event(self, state: Mapping[str, Any], event_type: str, record: Mapping[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_id": str(uuid.uuid4()),
            "sequence": int(state["last_sequence"]) + 1,
            "previous_digest": state["last_digest"],
            "occurred_at_utc": now.isoformat().replace("+00:00", "Z"),
            "budget_day": record.get("budget_day") or self._budget_day(now),
            "timezone": self.timezone_name,
            "event_type": event_type,
            "spend_domain": self.spend_domain,
            "aggregate_scope": record.get("aggregate_scope", ""),
            "provider_scope": record.get("provider_scope", ""),
            "route_scope": record.get("route_scope", ""),
            "model_scope": record.get("model_scope"),
            "provider_account_alias": record.get("provider_account_alias", ""),
            "provider": record.get("provider", ""),
            "model": record.get("model", ""),
            "service": record.get("service", ""),
            "environment": record.get("environment", ""),
            "route": record.get("route", ""),
            "policy_version": record.get("policy_version", ""),
            "caller_id": record.get("caller_id", ""),
            "correlation_id": record.get("correlation_id", ""),
            "idempotency_key_hash": record.get("idempotency_key_hash", ""),
            "reservation_id": record.get("reservation_id", ""),
            "provider_request_id_hash": record.get("provider_request_id_hash"),
            "reserved_calls": int(record.get("reserved_calls", 0)),
            "reserved_micro_usd": int(record.get("reserved_micro_usd", 0)),
            "actual_calls": int(record.get("actual_calls", 0)),
            "actual_micro_usd": int(record.get("actual_micro_usd", 0)),
            "input_tokens": int(record.get("input_tokens", 0)),
            "output_tokens": int(record.get("output_tokens", 0)),
            "outcome": record.get("outcome", ""),
            "reason": record.get("reason", ""),
            "attempt_number": int(record.get("attempt_number", 0)),
            "price_card_version": record.get("price_card_version", ""),
        }
        event["digest"] = _digest_mapping(event)
        return event

    def _append_event(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        line = _canonical_json(event) + b"\n"
        try:
            fd = os.open(self.journal_path, os.O_WRONLY | os.O_APPEND)
            try:
                written = os.write(fd, line)
                if written != len(line):
                    raise OSError("short journal append")
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception as exc:
            raise LedgerUnavailable("durable journal append failed") from exc
        self._apply_event(state, event)
        state["last_sequence"] = event["sequence"]
        state["last_digest"] = event["digest"]

    @staticmethod
    def _usage_for_scope(state: Mapping[str, Any], day: str, scope: str) -> tuple[int, int]:
        settled = state["totals"].get(day, {}).get(scope, {"calls": 0, "micro_usd": 0})
        calls = int(settled["calls"])
        money = int(settled["micro_usd"])
        for reservation in state["live_reservations"].values():
            if reservation["budget_day"] == day and scope in FileLedger._event_scopes(reservation):
                calls += int(reservation["reserved_calls"])
                money += int(reservation["reserved_micro_usd"])
        return calls, money

    def reserve(self, spec: ReservationSpec) -> LedgerReservation:
        if spec.spend_domain != self.spend_domain:
            raise LedgerUnavailable("reservation spend domain mismatch")
        if spec.price_card_version not in self.price_card_versions:
            raise LedgerUnavailable("unknown price-card version")
        if self.allowed_provider_accounts is not None and (
            spec.provider_account_alias not in self.allowed_provider_accounts
        ):
            raise LedgerUnavailable("unexpected provider account alias")
        with self._exclusive():
            _, state = self._load_state()
            if state["overage_blocked"]:
                raise BudgetDenied("spend domain is blocked after an overage")
            prior = state["idempotency"].get(spec.idempotency_key_hash)
            if prior is not None:
                raise IdempotencyConflict(str(prior.get("status", "unknown")))
            day = self._budget_day()
            for scope in spec.scopes:
                cap = self.caps.get(scope)
                if cap is None:
                    raise LedgerUnavailable("required cap scope is missing")
                used_calls, used_money = self._usage_for_scope(state, day, scope)
                if used_calls + 1 > cap.calls or used_money + spec.reserved_micro_usd > cap.micro_usd:
                    raise BudgetDenied("daily cap would be exceeded", details={"scope": scope})
            reservation_id = str(uuid.uuid4())
            record = {
                "budget_day": day,
                "aggregate_scope": spec.aggregate_scope,
                "provider_scope": spec.provider_scope,
                "route_scope": spec.route_scope,
                "model_scope": spec.model_scope,
                "provider_account_alias": spec.provider_account_alias,
                "provider": spec.provider,
                "model": spec.model,
                "service": spec.service,
                "environment": spec.environment,
                "route": spec.route,
                "policy_version": spec.policy_version,
                "caller_id": spec.caller_id,
                "correlation_id": spec.correlation_id_hash,
                "idempotency_key_hash": spec.idempotency_key_hash,
                "reservation_id": reservation_id,
                "reserved_calls": 1,
                "reserved_micro_usd": spec.reserved_micro_usd,
                "actual_calls": 0,
                "actual_micro_usd": 0,
                "attempt_number": spec.attempt_number,
                "price_card_version": spec.price_card_version,
                "outcome": "reserved",
            }
            event = self._base_event(state, "RESERVE", record)
            self._append_event(state, event)
            return LedgerReservation(
                reservation_id=reservation_id,
                budget_day=day,
                reserved_calls=1,
                reserved_micro_usd=spec.reserved_micro_usd,
                scopes=spec.scopes,
                idempotency_key_hash=spec.idempotency_key_hash,
            )

    def _live_record(self, state: Mapping[str, Any], reservation_id: str) -> dict[str, Any]:
        record = state["live_reservations"].get(reservation_id)
        if record is None:
            raise LedgerUnavailable("reservation is not live")
        return dict(record)

    def settle(
        self,
        reservation_id: str,
        *,
        actual_micro_usd: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        provider_request_id_hash: str | None = None,
        outcome: str = "success",
    ) -> None:
        values = (actual_micro_usd, input_tokens, output_tokens)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise LedgerUnavailable("settlement values must be non-negative integers")
        with self._exclusive():
            _, state = self._load_state()
            record = self._live_record(state, reservation_id)
            record.update(
                actual_calls=1,
                actual_micro_usd=actual_micro_usd,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_request_id_hash=provider_request_id_hash,
                outcome=outcome,
            )
            event = self._base_event(state, "SETTLE", record)
            self._append_event(state, event)
            if actual_micro_usd > int(record["reserved_micro_usd"]):
                overage = dict(record)
                overage["reason"] = "actual_charge_exceeded_reservation"
                overage_event = self._base_event(state, "OVERAGE", overage)
                self._append_event(state, overage_event)

    def release(self, reservation_id: str, *, reason: str) -> None:
        with self._exclusive():
            _, state = self._load_state()
            record = self._live_record(state, reservation_id)
            record.update(reason=reason, outcome="released")
            event = self._base_event(state, "RELEASE", record)
            self._append_event(state, event)

    def settle_unknown(
        self,
        reservation_id: str,
        *,
        reason: str,
        provider_request_id_hash: str | None = None,
    ) -> None:
        with self._exclusive():
            _, state = self._load_state()
            record = self._live_record(state, reservation_id)
            record.update(
                actual_calls=record["reserved_calls"],
                actual_micro_usd=record["reserved_micro_usd"],
                reason=reason,
                outcome="unknown",
                provider_request_id_hash=provider_request_id_hash,
            )
            event = self._base_event(state, "SETTLE_UNKNOWN", record)
            self._append_event(state, event)

    def remaining_budget(self, scope: str, *, caller: CallerContext) -> BudgetRemaining:
        validate_identifier(scope, "scope")
        cap = self.caps.get(scope)
        if cap is None:
            raise LedgerUnavailable("unknown cap scope")
        with self._exclusive():
            _, state = self._load_state()
            day = self._budget_day()
            calls, money = self._usage_for_scope(state, day, scope)
            return BudgetRemaining(
                scope=scope,
                budget_day=day,
                calls_remaining=max(0, cap.calls - calls),
                micro_usd_remaining=max(0, cap.micro_usd - money),
                calls_limit=cap.calls,
                micro_usd_limit=cap.micro_usd,
            )

    def unresolved_reservations(self) -> tuple[dict[str, Any], ...]:
        """Return privacy-safe metadata for reconciliation jobs."""

        with self._exclusive():
            _, state = self._load_state()
            fields = {
                "reservation_id",
                "provider",
                "model",
                "budget_day",
                "reserved_calls",
                "reserved_micro_usd",
                "idempotency_key_hash",
                "provider_request_id_hash",
            }
            return tuple(
                {key: value for key, value in record.items() if key in fields}
                for record in state["live_reservations"].values()
            )

    def recover(
        self,
        reservation_id: str,
        *,
        evidence_hash: str,
        actual_micro_usd: int | None = None,
        release: bool = False,
    ) -> None:
        """Append an evidence-linked operator resolution; old events stay immutable."""

        if len(evidence_hash) != 64 or any(ch not in "0123456789abcdef" for ch in evidence_hash):
            raise LedgerUnavailable("recovery requires a sha256 evidence hash")
        if release == (actual_micro_usd is not None):
            raise LedgerUnavailable("recovery must either release or settle")
        with self._exclusive():
            _, state = self._load_state()
            record = self._live_record(state, reservation_id)
            if release:
                record.update(outcome="released", reason=f"operator_evidence:{evidence_hash}")
            else:
                assert actual_micro_usd is not None
                if not isinstance(actual_micro_usd, int) or actual_micro_usd < 0:
                    raise LedgerUnavailable("recovery charge must be integer micro-USD")
                record.update(
                    outcome="settled",
                    reason=f"operator_evidence:{evidence_hash}",
                    actual_calls=1,
                    actual_micro_usd=actual_micro_usd,
                )
            event = self._base_event(state, "RECOVERY", record)
            self._append_event(state, event)

    @staticmethod
    def _compress_zstd(data: bytes) -> bytes:
        library_name = ctypes.util.find_library("zstd")
        if not library_name:
            raise LedgerUnavailable("libzstd is required for immutable archive creation")
        lib = ctypes.CDLL(library_name)
        lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        lib.ZSTD_compressBound.restype = ctypes.c_size_t
        lib.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.ZSTD_compress.restype = ctypes.c_size_t
        lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        lib.ZSTD_isError.restype = ctypes.c_uint
        bound = int(lib.ZSTD_compressBound(len(data)))
        destination = ctypes.create_string_buffer(bound)
        source = ctypes.create_string_buffer(data)
        size = int(lib.ZSTD_compress(destination, bound, source, len(data), 3))
        if lib.ZSTD_isError(size):
            raise LedgerUnavailable("zstd archive compression failed")
        return destination.raw[:size]

    def _snapshot_from_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "spend_domain": self.spend_domain,
            "timezone": self.timezone_name,
            "checkpoint_sequence": state["last_sequence"],
            "checkpoint_digest": state["last_digest"],
            "totals": state["totals"],
            "live_reservations": state["live_reservations"],
            "idempotency": state["idempotency"],
            "overage_blocked": state["overage_blocked"],
            "price_card_versions": sorted(self.price_card_versions),
        }

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _atomic_write_snapshot(self, snapshot: dict[str, Any]) -> None:
        snapshot = dict(snapshot)
        snapshot["digest"] = _digest_mapping(snapshot)
        temporary = self.directory / f".snapshot.{uuid.uuid4().hex}.tmp"
        try:
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                payload = _canonical_json(snapshot) + b"\n"
                written = os.write(fd, payload)
                if written != len(payload):
                    raise OSError("short snapshot write")
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temporary, self.snapshot_path)
            self._fsync_directory(self.directory)
        except Exception as exc:
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass
            raise LedgerUnavailable("atomic snapshot update failed") from exc

    def compact(self, *, now: datetime | None = None) -> Path:
        """Checkpoint, snapshot, and rotate the active journal atomically under the ledger lock."""

        with self._exclusive():
            _, state = self._load_state()
            checkpoint = self._base_event(state, "CHECKPOINT", {"outcome": "checkpoint"})
            self._append_event(state, checkpoint)
            self._atomic_write_snapshot(self._snapshot_from_state(state))
            month = (now or datetime.now(timezone.utc)).astimezone(self.timezone).strftime("%Y-%m")
            archive_path = self.archive_dir / f"{month}.jsonl.zst"
            if archive_path.exists():
                raise LedgerUnavailable("monthly immutable archive already exists")
            try:
                raw_journal = self.journal_path.read_bytes()
                compressed = self._compress_zstd(raw_journal)
                archive_fd = os.open(archive_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400)
                try:
                    written = os.write(archive_fd, compressed)
                    if written != len(compressed):
                        raise OSError("short archive write")
                    os.fsync(archive_fd)
                finally:
                    os.close(archive_fd)
                self._fsync_directory(self.archive_dir)
                rotated = self.directory / f".journal.{uuid.uuid4().hex}.closed"
                os.replace(self.journal_path, rotated)
                journal_fd = os.open(self.journal_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(journal_fd)
                self._fsync_directory(self.directory)
                rotated.unlink()
            except Exception as exc:
                raise LedgerUnavailable("journal compaction failed") from exc

            # Snapshot represents the closed checkpoint. The new active journal
            # starts with a chain-linked checkpoint event.
            _, current = self._load_state()
            new_checkpoint = self._base_event(current, "CHECKPOINT", {"outcome": "journal_opened"})
            self._append_event(current, new_checkpoint)
            return archive_path

    def diagnose(self) -> dict[str, Any]:
        """Read-only integrity diagnostics; it never repairs or edits evidence."""

        report: dict[str, Any] = {
            "spend_domain": self.spend_domain,
            "timezone": self.timezone_name,
            "healthy": False,
            "error_code": None,
        }
        try:
            with self._exclusive():
                _, state = self._load_state()
            report.update(
                healthy=True,
                last_sequence=state["last_sequence"],
                live_reservations=len(state["live_reservations"]),
                overage_blocked=state["overage_blocked"],
            )
        except (LedgerUnavailable, BudgetDenied) as exc:
            report["error_code"] = exc.code
        return report
