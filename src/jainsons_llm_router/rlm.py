"""Fail-soft large-data detection ported from the harness RLM router.

This module is a dependency-free adaptation of the harness's
``harness_rlm_router.py``. The canonical and most-current detector lives in
the harness; if it changes, this file must be re-ported by a human. It is not
auto-synced like ``policies/harness_derived.py`` via ``sync_from_harness.py``.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

_LARGE_FILE_BYTES = 200 * 1024
_LARGE_TEXT_BYTES = 1024 * 1024
_LARGE_FILE_COUNT = 50
_LARGE_TABLE_ROWS = 5000
_DATA_SUFFIXES = frozenset({".csv", ".xlsx", ".tsv", ".db", ".sqlite", ".sqlite3"})
_TEXT_SUFFIXES = frozenset(
    {
        ".cfg",
        ".conf",
        ".csv",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".rst",
        ".sql",
        ".tsv",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_ACTION_RE = re.compile(
    r"\b(?:write|writing|change|changing|delete|deleting|update|updating|"
    r"edit|editing|modify|modifying|create|creating|remove|removing)\b",
    re.IGNORECASE,
)
_VISION_RE = re.compile(
    r"\b(?:vision|image|render(?:ing)?|screenshot|photo|picture|illustration|visual)\b",
    re.IGNORECASE,
)
_KEYWORD_RE = re.compile(
    r"\b(?:all products|catalogue|catalog|boq|price list|entire|across all|"
    r"dead stock|every sku|every order|every product)\b",
    re.IGNORECASE,
)
_QUOTED_TOKEN_RE = re.compile(r"(?:\"([^\"]+)\"|'([^']+)'|`([^`]+)`)")
_PATH_TOKEN_RE = re.compile(
    r"(?:~[\\/][^\s'\"`]+|(?:\.{1,2}[\\/]|/)[^\s'\"`]+|"
    r"[A-Za-z]:[\\/][^\s'\"`]+|(?:[\w.-]+[\\/])+(?:[\w*?\[\].-]+)?|"
    r"(?:[*?][\w*?\[\].-]*\.[A-Za-z0-9]{1,16})|[\w.-]+\.[A-Za-z0-9]{1,16})"
)


def _harness_repo() -> Path:
    return Path.cwd()


def _clean_token(token: str) -> str:
    return token.strip().rstrip(".,;:!?)]}>'\"")


def _looks_pathish(token: str) -> bool:
    suffix = Path(token).suffix.lower()
    if (
        suffix
        or "/" in token
        or "\\" in token
        or token.startswith("~")
        or glob.has_magic(token)
    ):
        return True
    try:
        return Path(_expanded_token(token)).exists()
    except Exception:  # noqa: BLE001 - an unresolved quoted path is not a signal
        return False


def _path_tokens(task: str) -> list[str]:
    try:
        tokens: list[str] = []
        for match in _QUOTED_TOKEN_RE.finditer(task):
            token = _clean_token(next(part for part in match.groups() if part is not None))
            if token and _looks_pathish(token):
                tokens.append(token)
        tokens.extend(_clean_token(match.group(0)) for match in _PATH_TOKEN_RE.finditer(task))
        return list(dict.fromkeys(token for token in tokens if token))
    except Exception:  # noqa: BLE001 - token parsing is advisory only
        return []


def _expanded_token(token: str) -> str:
    expanded = os.path.expanduser(token)
    if not os.path.isabs(expanded):
        expanded = os.path.join(str(_harness_repo()), expanded)
    return expanded


def _targets_for(token: str) -> list[Path]:
    try:
        expanded = _expanded_token(token)
        if glob.has_magic(expanded):
            matches: list[Path] = []
            for match in glob.iglob(expanded, recursive=True):
                matches.append(Path(match))
                if len(matches) > _LARGE_FILE_COUNT:
                    break
            return matches
        path = Path(expanded)
        return [path] if path.exists() else []
    except Exception:  # noqa: BLE001 - inaccessible targets are ignored fail-soft
        return []


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_SUFFIXES


def _format_bytes(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MiB"
    return f"{size / 1024:.1f} KiB"


def _db_large_table(path: Path) -> str | None:
    """Return a table name over the threshold while keeping SQLite strictly read-only."""
    try:
        uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=1) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
            for (table_name,) in rows:
                quoted_name = str(table_name).replace('"', '""')
                query = f'SELECT 1 FROM "{quoted_name}" LIMIT 1 OFFSET {_LARGE_TABLE_ROWS}'  # nosec B608
                if connection.execute(query).fetchone() is not None:
                    return str(table_name)
    except Exception:  # noqa: BLE001 - malformed or locked databases are not a signal
        return None
    return None


def _walk_files(target: Path):
    try:
        if target.is_file():
            yield target
            return
        if not target.is_dir():
            return
        for root, _, names in os.walk(target, followlinks=False):
            for name in names:
                candidate = Path(root, name)
                if candidate.is_file():
                    yield candidate
    except Exception:  # noqa: BLE001 - inaccessible trees are ignored fail-soft
        return


def _scan_token(token: str) -> dict[str, object]:
    """Collect only enough filesystem evidence to decide whether the token is large."""
    result: dict[str, object] = {
        "db_table": None,
        "file_count": 0,
        "large_file": None,
        "paths": [],
        "text_bytes": 0,
        "trivial": False,
    }
    try:
        targets = _targets_for(token)
        result["paths"] = [str(path.resolve()) for path in targets[:8]]
        if not targets:
            return result
        seen: set[str] = set()
        for target in targets:
            for path in _walk_files(target):
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                size = path.stat().st_size
                result["file_count"] = int(result["file_count"]) + 1
                if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
                    table_name = _db_large_table(path)
                    if table_name:
                        result["db_table"] = table_name
                        return result
                if size > _LARGE_FILE_BYTES:
                    result["large_file"] = (str(path.resolve()), size)
                    return result
                if _is_text_file(path):
                    result["text_bytes"] = int(result["text_bytes"]) + size
                if (
                    int(result["file_count"]) > _LARGE_FILE_COUNT
                    or int(result["text_bytes"]) > _LARGE_TEXT_BYTES
                ):
                    return result
        result["trivial"] = True
    except Exception:  # noqa: BLE001 - inspection must remain advisory
        return result
    return result


def _signal(paths: list[str], reason: str, approx: str) -> dict[str, object]:
    return {"paths": list(dict.fromkeys(paths)), "reason": reason, "approx": approx}


def rlm_large_data_signal(task: str) -> dict | None:
    """Return fail-soft evidence that a read-only inspector would avoid context overload."""
    try:
        if not isinstance(task, str):
            return None
        if "--no-rlm" in task.lower() or _ACTION_RE.search(task) or _VISION_RE.search(task):
            return None

        tokens = _path_tokens(task)
        referenced_paths: list[str] = []
        data_states: list[bool] = []
        for token in tokens:
            scan = _scan_token(token)
            paths = [str(path) for path in scan["paths"]]
            referenced_paths.extend(paths or [token])
            data_token = Path(token).suffix.lower() in _DATA_SUFFIXES
            if data_token:
                data_states.append(bool(scan["trivial"]) if paths else False)
            db_table = scan["db_table"]
            if db_table:
                return _signal(
                    paths or [token],
                    f"SQLite table '{db_table}' exceeds {_LARGE_TABLE_ROWS} rows",
                    f"> {_LARGE_TABLE_ROWS} rows",
                )
            large_file = scan["large_file"]
            if large_file:
                large_path, size = large_file
                return _signal(
                    [str(large_path)],
                    f"referenced file exceeds {_LARGE_FILE_BYTES // 1024} KiB",
                    _format_bytes(int(size)),
                )
            file_count = int(scan["file_count"])
            text_bytes = int(scan["text_bytes"])
            if file_count > _LARGE_FILE_COUNT:
                return _signal(
                    paths or [token],
                    f"referenced directory or glob has more than {_LARGE_FILE_COUNT} files",
                    f"{file_count} files",
                )
            if text_bytes > _LARGE_TEXT_BYTES:
                return _signal(
                    paths or [token],
                    "referenced directory or glob has more than 1 MiB of text",
                    _format_bytes(text_bytes),
                )

        only_trivial_data = bool(data_states) and all(data_states)
        if only_trivial_data:
            return None
        keyword_match = _KEYWORD_RE.search(task)
        if keyword_match:
            return _signal(
                referenced_paths,
                f"'{keyword_match.group(0)}' indicates a catalogue-scale dataset",
                "keyword signal",
            )
        for token in tokens:
            suffix = Path(token).suffix.lower()
            if suffix in _DATA_SUFFIXES:
                return _signal(
                    referenced_paths or [token],
                    f"referenced {suffix} data path",
                    "data path reference",
                )
    except Exception:  # noqa: BLE001 - this public detector must never raise
        return None
    return None


def _append_auto_log(task: str, signal: dict) -> None:
    try:
        log_path = os.path.expanduser("~/.claude/logs/rlm_auto.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "task_snippet": task[:500],
            "reason": signal.get("reason", "large data"),
            "paths": signal.get("paths", []),
            "source": "llm_router",
        }
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - read-only log directories are expected
        return


def maybe_inject_rlm_nudge(task: str) -> str:
    """Add optional large-data guidance without affecting routing or failure behavior."""
    try:
        if os.environ.get("HARNESS_RLM_AUTO", "1") == "0":  # active for the team by default; kill-switch = set to "0"
            return task
        signal = rlm_large_data_signal(task)
        if not signal:
            return task
        paths = signal.get("paths", [])
        reason = signal.get("reason", "large data")
        inspector_path = Path.home() / ".claude" / "skills" / "harness-offload"
        nudge = (
            f"[LARGE-DATA MODE] This task involves large data ({reason}). Do NOT load it all into "
            "context. Use the read-only context inspector: add "
            f"{inspector_path} to sys.path then "
            "`from harness_context_inspector import InspectorSession`; create "
            f"InspectorSession({paths!r}) and fetch ONLY the slices you need via peek / grep / sql "
            "(SELECT-only) / find_gaps. Fetching only what you need avoids context rot.\n\n"
        )
        _append_auto_log(task, signal)
        return nudge + task
    except Exception:  # noqa: BLE001 - the nudge must never affect worker execution
        return task
