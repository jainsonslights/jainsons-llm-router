from __future__ import annotations

import inspect
import json

from jainsons_llm_router import client
from jainsons_llm_router import rlm


def _large_file(tmp_path):
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * (rlm._LARGE_FILE_BYTES + 1))
    return path


def test_large_data_signal_ignores_short_data_free_prompt() -> None:
    prompt = "Summarize this paragraph: a short piece of ordinary prose."

    assert rlm.rlm_large_data_signal(prompt) is None


def test_large_data_signal_detects_file_over_threshold(tmp_path) -> None:
    path = _large_file(tmp_path)

    signal = rlm.rlm_large_data_signal(f"Summarize {path}")

    assert signal is not None
    assert str(path) in signal["paths"]


def test_nudge_is_prepended_and_uses_call_time_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _large_file(tmp_path)
    prompt = f"Summarize {path}"

    result = rlm.maybe_inject_rlm_nudge(prompt)

    assert result.startswith("[LARGE-DATA MODE]")
    assert prompt in result
    assert str(tmp_path) in result


def test_nudge_kill_switch_returns_prompt_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HARNESS_RLM_AUTO", "0")
    path = _large_file(tmp_path)
    prompt = f"Summarize {path}"

    assert rlm.maybe_inject_rlm_nudge(prompt) == prompt


def test_nudge_returns_small_prompt_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    prompt = "Summarize this paragraph: a short piece of ordinary prose."

    assert rlm.maybe_inject_rlm_nudge(prompt) == prompt


def test_nudge_log_identifies_router_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _large_file(tmp_path)

    rlm.maybe_inject_rlm_nudge(f"Summarize {path}")

    log_path = tmp_path / ".claude" / "logs" / "rlm_auto.jsonl"
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["source"] == "llm_router"


def test_client_rlm_wrapper_fails_soft(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def broken_nudge(prompt: str) -> str:
        raise RuntimeError("broken RLM")

    monkeypatch.setattr(rlm, "maybe_inject_rlm_nudge", broken_nudge)

    assert client._maybe_apply_rlm("hello") == "hello"


def test_complete_text_applies_rlm_before_lane_resolution() -> None:
    source = inspect.getsource(client.complete_text)

    assert source.index("_maybe_apply_rlm(") < source.index("_lane_candidates(")
