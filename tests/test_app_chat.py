from __future__ import annotations

import asyncio
import concurrent.futures
import http.client
import http.server
import io
import json
import os
import select
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from jainsons_llm_router import BudgetExhausted, RouteUnavailable, acomplete_chat, complete_chat, set_alert_sink
from jainsons_llm_router import app_chat
from jainsons_llm_router import runtime_policy
from jainsons_llm_router.runtime_policy import canonical_app_policy_sha256


FIXTURE = Path(__file__).parent / "fixtures" / "promoted_app_policy.json"


class FakeResponse:
    def __init__(self, payload: bytes = b"", status: int = 200, *, delay: float = 0) -> None:
        self.payload = payload
        self.status = status
        self.delay = delay
        self.closed = False

    def read1(self, _size: int) -> bytes:
        if self.closed:
            return b""
        if self.delay:
            time.sleep(self.delay)
        self.closed = True
        return self.payload

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    def __init__(self, responses: list[FakeResponse | BaseException]) -> None:
        self.responses = responses
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append((request, timeout))
        response_or_error = self.responses.pop(0)
        if isinstance(response_or_error, BaseException):
            raise response_or_error
        return response_or_error


def response(
    text: str = "answer",
    *,
    model: str = "deepseek-flash",
    usage: dict[str, int] | None = None,
) -> FakeResponse:
    payload = {"model": model, "choices": [{"message": {"content": text}}]}
    if usage is not None:
        payload["usage"] = usage
    return FakeResponse(json.dumps(payload).encode())


def policy_file(tmp_path: Path, *, mutate=None) -> Path:
    policy = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if mutate:
        mutate(policy)
    policy["app_policy_sha256"] = canonical_app_policy_sha256(policy["app_backends"], policy["app_lanes"])
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    return path


def spend_state(tmp_path: Path) -> dict[str, object]:
    files = list((tmp_path / "spend").glob("chat_fast-*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_primary_http_failure_falls_back_and_payload_is_protected(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "open-secret")
    opener = FakeOpener([FakeResponse(status=500), response("fallback", model="z-ai/glm-5.3-flash")])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)

    result = complete_chat(
        [{"role": "user", "content": "private prompt"}],
        max_output_tokens=17,
        policy_path=path,
    )

    assert result.text == "fallback"
    assert result.backend == "glm-openrouter-flash"
    assert len(opener.requests) == 2
    first_request, _ = opener.requests[0]
    second_request, _ = opener.requests[1]
    assert first_request.full_url.startswith("https://api.deepseek.com/")
    assert second_request.full_url.startswith("https://openrouter.ai/")
    assert first_request.headers["Authorization"] == "Bearer deep-secret"
    assert second_request.headers["Authorization"] == "Bearer open-secret"
    body = json.loads(second_request.data)
    assert body["model"] == "z-ai/glm-5.3-flash"
    assert body["max_tokens"] == 17
    assert body["messages"] == [{"role": "user", "content": "private prompt"}]
    assert body["provider"] if "provider" in body else True
    assert "model" not in body.get("provider", {})


def test_read_waits_for_the_remaining_attempt_deadline(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")

    class PausedResponse(FakeResponse):
        def settimeout(self, value):
            self.timeout = value

        def read1(self, size):
            assert self.timeout > 1.5
            time.sleep(1.5)
            return super().read1(size)

    opener = FakeOpener([PausedResponse(response("after-pause").payload)])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)
    assert complete_chat([{"role": "user", "content": "hello"}], deadline_seconds=10, policy_path=path).text == "after-pause"


def test_backend_minimum_request_tokens_controls_payload(tmp_path, monkeypatch):
    def adjust(policy):
        policy["app_backends"]["deepseek-direct-flash"]["min_request_max_tokens"] = 2000

    path = policy_file(tmp_path, mutate=adjust)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    opener = FakeOpener([response("ok")])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)
    complete_chat([{"role": "user", "content": "hello"}], max_output_tokens=120, policy_path=path)
    assert json.loads(opener.requests[0][0].data)["max_tokens"] == 2000


def test_success_settles_with_provider_reported_usage(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    opener = FakeOpener([
        response("metered", usage={"prompt_tokens": 10, "completion_tokens": 20}),
    ])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)

    result = complete_chat([{"role": "user", "content": "hello"}], policy_path=path)

    state = spend_state(tmp_path)
    assert state["spent_usd"] == 0.000027
    assert state["reservations"] == {}
    assert result.cost_usd == 0.000027


def test_post_send_timeout_settles_unknown_but_pre_send_refusal_releases(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "open-secret")

    class TimedOutResponse(FakeResponse):
        def read1(self, _size):
            raise TimeoutError

    timeout_opener = FakeOpener([
        TimedOutResponse(),
        response("fallback", model="z-ai/glm-5.3-flash"),
    ])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: timeout_opener)
    assert complete_chat([{"role": "user", "content": "hello"}], policy_path=path).text == "fallback"
    timeout_state = spend_state(tmp_path)
    assert timeout_state["spent_usd"] > 0
    assert timeout_state["reservations"] == {}

    second_dir = tmp_path / "refused"
    second_dir.mkdir()
    refused_path = policy_file(second_dir)
    refused_opener = FakeOpener([
        urllib.error.URLError(ConnectionRefusedError()),
        response("fallback", model="z-ai/glm-5.3-flash"),
    ])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: refused_opener)
    assert complete_chat([{"role": "user", "content": "hello"}], policy_path=refused_path).text == "fallback"
    refusal_state = spend_state(second_dir)
    assert refusal_state["spent_usd"] > 0
    assert refusal_state["reservations"] == {}


@pytest.mark.parametrize("failure", [TimeoutError(), OSError("response wait failed")])
def test_open_phase_ambiguous_failures_settle_unknown(tmp_path, monkeypatch, failure):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "open-secret")
    opener = FakeOpener([failure, response("fallback", model="z-ai/glm-5.3-flash")])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)
    assert complete_chat([{"role": "user", "content": "hello"}], policy_path=path).text == "fallback"
    assert spend_state(tmp_path)["spent_usd"] > 0


@pytest.mark.parametrize(
    "reason",
    [ConnectionRefusedError(), socket.gaierror(socket.EAI_NONAME, "not found"), ssl.SSLError("handshake")],
)
def test_definite_pre_send_url_errors_release_reservation(tmp_path, monkeypatch, reason):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "open-secret")
    opener = FakeOpener([
        urllib.error.URLError(reason),
        response("fallback", model="z-ai/glm-5.3-flash"),
    ])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)
    result = complete_chat([{"role": "user", "content": "hello"}], policy_path=path)
    assert result.text == "fallback"
    assert spend_state(tmp_path)["spent_usd"] == result.cost_usd


def test_trickling_response_hits_primary_wall_clock_share_then_fallback(tmp_path, monkeypatch):
    def adjust(policy):
        policy["app_lanes"]["chat_fast"]["primary_share"] = 0.2

    path = policy_file(tmp_path, mutate=adjust)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "open-secret")

    class TricklingResponse(FakeResponse):
        def read1(self, _size):
            time.sleep(0.03)
            return b"\n"

    opener = FakeOpener([TricklingResponse(), response("fallback")])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)
    started = time.monotonic()
    result = complete_chat([{"role": "user", "content": "hello"}], deadline_seconds=5, policy_path=path)
    elapsed = time.monotonic() - started
    assert result.text == "fallback"
    assert result.backend == "glm-openrouter-flash"
    assert 1 <= elapsed < 5


def test_budget_exhaustion_happens_before_network(tmp_path, monkeypatch):
    def adjust(policy):
        policy["app_lanes"]["chat_fast"]["daily_cap_usd"] = 0.000001

    path = policy_file(tmp_path, mutate=adjust)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    opener = FakeOpener([response()])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)
    with pytest.raises(BudgetExhausted):
        complete_chat([{"role": "user", "content": "hello"}], policy_path=path)
    assert opener.requests == []


def test_model_mismatch_alert_is_redacted_and_thresholds_are_durable(tmp_path, monkeypatch):
    def adjust(policy):
        lane = policy["app_lanes"]["chat_fast"]
        lane["daily_cap_usd"] = 0.0001
        for field in ("input_usd_per_million", "output_usd_per_million"):
            policy["app_backends"]["deepseek-direct-flash"]["price_card"][field] = 1

    path = policy_file(tmp_path, mutate=adjust)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    seen = []
    set_alert_sink(seen.append)
    try:
        for day in ("2026-09-14", "2026-09-15"):
            monkeypatch.setattr(app_chat.AppSpend, "_day", lambda self: day)
            for output_tokens in (80, 10):
                # Fresh policy and ledger objects must use the persisted budget/dedupe state.
                runtime_policy._RUNTIMES.pop(path, None)
                opener = FakeOpener([response(
                    "answer", model="other-model",
                    usage={"prompt_tokens": 5, "completion_tokens": output_tokens},
                )])
                monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)
                result = complete_chat(
                    [{"role": "user", "content": "private prompt"}],
                    max_output_tokens=output_tokens, policy_path=path,
                )
                assert result.text == "answer"
                assert result.reply_model == "other-model"
            # Repeated cap denials from fresh objects must not duplicate either alert.
            for _ in range(2):
                runtime_policy._RUNTIMES.pop(path, None)
                with pytest.raises(BudgetExhausted):
                    complete_chat([{"role": "user", "content": "private prompt"}], policy_path=path)
            reasons = [event["reason"] for event in seen]
            days_elapsed = 1 if day == "2026-09-14" else 2
            assert reasons.count("budget_80") == days_elapsed
            assert reasons.count("budget_100") == days_elapsed
        assert reasons.count("reply_model_mismatch") == 1
    finally:
        set_alert_sink(None)
    alerts = (tmp_path / "alerts.jsonl").read_text(encoding="utf-8")
    assert "private prompt" not in alerts
    assert "deep-secret" not in alerts


def test_async_calls_run_in_parallel(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    rendezvous = threading.Barrier(2, timeout=2)

    class SlowResponse(FakeResponse):
        def read1(self, size):
            if not self.closed:
                rendezvous.wait()  # Sequential execution cannot pass this barrier.
            return super().read1(size)

    opener = FakeOpener([
        SlowResponse(response(text).payload, delay=0.12) for text in ("one", "two")
    ])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)

    async def run():
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(heartbeat())
        try:
            results = await asyncio.wait_for(asyncio.gather(
                acomplete_chat([{"role": "user", "content": "one"}], policy_path=path),
                acomplete_chat([{"role": "user", "content": "two"}], policy_path=path),
            ), timeout=3)
            assert ticks >= 2  # The event loop remained responsive during both calls.
            return results
        finally:
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)

    # Own shutdown synchronously: this sandbox can lose cross-thread loop wakeups.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        loop = asyncio.new_event_loop()
        loop.set_default_executor(executor)
        try:
            results = loop.run_until_complete(run())
        finally:
            loop.close()
    assert {item.text for item in results} == {"one", "two"}


def test_both_backends_fail_raises_route_unavailable(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "open-secret")
    opener = FakeOpener([FakeResponse(status=500), FakeResponse(status=502)])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)

    with pytest.raises(RouteUnavailable, match="no application chat backend is available"):
        complete_chat([{"role": "user", "content": "hello"}], policy_path=path)
    assert len(opener.requests) == 2


def test_missing_key_skips_backend(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "open-secret")
    opener = FakeOpener([response("fallback-only", model="z-ai/glm-5.3-flash")])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)

    result = complete_chat([{"role": "user", "content": "hello"}], policy_path=path)
    assert result.text == "fallback-only"
    assert result.backend == "glm-openrouter-flash"
    assert len(opener.requests) == 1
    req, _ = opener.requests[0]
    assert req.full_url.startswith("https://openrouter.ai/")

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    opener.requests.clear()
    with pytest.raises(RouteUnavailable):
        complete_chat([{"role": "user", "content": "hello"}], policy_path=path)
    assert opener.requests == []


@pytest.mark.parametrize("protected_key", [
    "model", "messages", "max_tokens", "max_completion_tokens", "stream", "n", "tools", "tool_choice",
])
def test_payload_defaults_cannot_override_model_messages_or_max_tokens(tmp_path, monkeypatch, protected_key):
    def inject_override(policy):
        policy["app_backends"]["deepseek-direct-flash"]["transport"]["payload_defaults"] = {
            protected_key: "attacker-controlled",
        }

    path = policy_file(tmp_path, mutate=inject_override)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    opener = FakeOpener([])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)

    with pytest.raises(RouteUnavailable):
        complete_chat([{"role": "user", "content": "test message"}], policy_path=path)
    assert opener.requests == []


def test_key_only_sent_to_bound_host(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key-1234")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-5678")
    opener = FakeOpener([FakeResponse(status=500), response("ok", model="z-ai/glm-5.3-flash")])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)

    complete_chat([{"role": "user", "content": "hello"}], policy_path=path)
    assert len(opener.requests) == 2
    ds_req, _ = opener.requests[0]
    or_req, _ = opener.requests[1]
    assert ds_req.host == "api.deepseek.com"
    assert ds_req.headers["Authorization"] == "Bearer ds-key-1234"
    assert "or-key-5678" not in ds_req.headers.values()

    assert or_req.host == "openrouter.ai"
    assert or_req.headers["Authorization"] == "Bearer or-key-5678"
    assert "ds-key-1234" not in or_req.headers.values()


def test_no_prompt_keys_or_provider_bodies_in_exceptions_alerts_or_logs(tmp_path, monkeypatch, caplog, capsys):
    path = policy_file(tmp_path)
    prompt_text = "VERY_PRIVATE_CUSTOMER_PROMPT_98765"
    deep_secret = "ds-super-secret-key-9999"
    open_secret = "or-super-secret-key-8888"
    provider_leak = "PROVIDER_INTERNAL_ERROR_STACK_TRACE_12345"

    monkeypatch.setenv("DEEPSEEK_API_KEY", deep_secret)
    monkeypatch.setenv("OPENROUTER_API_KEY", open_secret)

    class ErrorResponse(FakeResponse):
        def __init__(self):
            super().__init__(provider_leak.encode(), status=500)

    opener = FakeOpener([ErrorResponse(), ErrorResponse()])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)

    seen_alerts = []
    set_alert_sink(seen_alerts.append)
    try:
        with pytest.raises(RouteUnavailable) as exc_info:
            complete_chat([{"role": "user", "content": prompt_text}], policy_path=path)
    finally:
        set_alert_sink(None)

    import traceback

    captured = capsys.readouterr()
    error_str = "".join(traceback.format_exception(exc_info.type, exc_info.value, exc_info.tb))
    error_str += caplog.text + captured.out + captured.err
    error_str += json.dumps(spend_state(tmp_path))
    assert prompt_text not in error_str
    assert deep_secret not in error_str
    assert open_secret not in error_str
    assert provider_leak not in error_str

    for alert in seen_alerts:
        alert_str = str(alert)
        assert prompt_text not in alert_str
        assert deep_secret not in alert_str
        assert open_secret not in alert_str
        assert provider_leak not in alert_str

    alerts_file = tmp_path / "alerts.jsonl"
    if alerts_file.exists():
        content = alerts_file.read_text(encoding="utf-8")
        assert prompt_text not in content
        assert deep_secret not in content
        assert open_secret not in content
        assert provider_leak not in content


def test_oversized_response_falls_back_to_next_backend(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "open-secret")

    oversized = b"x" * (2 * 1024 * 1024 + 10)
    opener = FakeOpener([FakeResponse(payload=oversized), response("recovered", model="z-ai/glm-5.3-flash")])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)

    result = complete_chat([{"role": "user", "content": "hello"}], policy_path=path)
    assert result.text == "recovered"
    assert result.backend == "glm-openrouter-flash"
    assert len(opener.requests) == 2


def test_parameter_clamping_and_validation(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    opener = FakeOpener([response("ok")])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)

    result = complete_chat(
        [{"role": "user", "content": "hi"}],
        max_output_tokens=999999,
        deadline_seconds=999999,
        policy_path=path,
    )
    assert result.text == "ok"
    req, timeout = opener.requests[0]
    body = json.loads(req.data)
    assert body["max_tokens"] == 1300
    assert 54 < timeout <= 55

    with pytest.raises(app_chat.ConfigurationError):
        complete_chat([{"role": "hacker", "content": "hi"}], policy_path=path)
    with pytest.raises(app_chat.ConfigurationError):
        complete_chat([{"role": "user", "content": ""}], policy_path=path)
    with pytest.raises(app_chat.ConfigurationError):
        complete_chat([{"role": "user", "content": "x" * 70000}], policy_path=path)


def test_primary_share_only_applies_when_a_later_backend_has_a_key(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    opener = FakeOpener([response("only-usable")])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)
    complete_chat([{"role": "user", "content": "hello"}], deadline_seconds=10, policy_path=path)
    assert 9.5 < opener.requests[0][1] <= 10

    single_dir = tmp_path / "single"
    single_dir.mkdir()

    def one_backend(policy):
        policy["app_lanes"]["chat_fast"]["backend_chain"] = ["deepseek-direct-flash"]

    single_path = policy_file(single_dir, mutate=one_backend)
    single_opener = FakeOpener([response("single")])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: single_opener)
    complete_chat([{"role": "user", "content": "hello"}], deadline_seconds=1, policy_path=single_path)
    assert 0.8 < single_opener.requests[0][1] <= 1


class PipeSocket:
    """Socket-shaped local byte stream for urllib/http.server under seccomp."""

    def __init__(self, read_fd, write_fd):
        self.read_fd, self.write_fd = read_fd, write_fd
        self.timeout = None
        self.closed = False

    @classmethod
    def pair(cls):
        a_read, b_write = os.pipe()
        b_read, a_write = os.pipe()
        return cls(a_read, a_write), cls(b_read, b_write)

    def settimeout(self, timeout):
        self.timeout = timeout

    def sendall(self, data):
        while data:
            data = data[os.write(self.write_fd, data):]

    def makefile(self, mode, buffering=-1):
        assert mode == "rb"
        peer = self

        class Reader(io.RawIOBase):
            def __init__(self):
                self.fd = os.dup(peer.read_fd)
                self._sock = peer  # Same timeout discovery path as a socket file.

            def readable(self):
                return True

            def readinto(self, buffer):
                ready, _, _ = select.select([self.fd], [], [], peer.timeout)
                if not ready:
                    raise TimeoutError
                data = os.read(self.fd, len(buffer))
                buffer[:len(data)] = data
                return len(data)

            def close(self):
                if not self.closed:
                    os.close(self.fd)
                super().close()

        return io.BufferedReader(Reader())

    def close(self):
        if not self.closed:
            self.closed = True
            os.close(self.read_fd)
            os.close(self.write_fd)


@pytest.fixture
def local_http(monkeypatch):
    """Serve HTTP on loopback, or OS pipes when the sandbox forbids sockets."""
    servers, threads, sockets = [], [], []
    requests = []
    build_opener = urllib.request.build_opener
    original_connect = http.client.HTTPConnection.connect

    def install(respond):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                requests.append((json.loads(body), self.headers.get("Authorization")))
                try:
                    respond(self)
                except (OSError, ValueError):
                    pass  # The deadline test deliberately closes the client mid-stream.

            def log_message(self, *args):
                pass

        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError:
            port = 18099

            def connect(connection):
                if connection.host != "127.0.0.1" or connection.port != port:
                    return original_connect(connection)
                client, peer = PipeSocket.pair()
                sockets.extend((client, peer))
                client.settimeout(connection.timeout)
                connection.sock = client

                def serve():
                    try:
                        Handler(peer, ("127.0.0.1", port), None)
                    finally:
                        peer.close()

                thread = threading.Thread(target=serve, daemon=True)
                threads.append(thread)
                thread.start()

            monkeypatch.setattr(http.client.HTTPConnection, "connect", connect)
        else:
            port = server.server_port
            servers.append(server)
            thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True)
            threads.append(thread)
            thread.start()

        class LocalRedirectHandler(urllib.request.BaseHandler):
            handler_order = 100  # Intercept before urllib's built-in HTTPS handler.

            def _open_local(self, req):
                local_request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/chat/completions",
                    data=req.data, headers=dict(req.header_items()), method=req.get_method(),
                )
                local_request.timeout = req.timeout
                return urllib.request.HTTPHandler().http_open(local_request)

            def http_open(self, req):
                return self._open_local(req)

            def https_open(self, req):
                return self._open_local(req)

        opener = build_opener(
            urllib.request.ProxyHandler({}), app_chat._NoRedirectHandler(), LocalRedirectHandler(),
        )
        monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)
        return requests

    yield install
    for server in servers:
        server.shutdown()
        server.server_close()
    for sock in sockets:
        sock.close()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive(), "local HTTP server failed to stop"


def send_local_answer(handler, text="from-local", model="deepseek-flash"):
    body = json.dumps({"model": model, "choices": [{"message": {"content": text}}]}).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def test_local_server_dispatch_or_skip(tmp_path, monkeypatch, local_http):
    requests = local_http(send_local_answer)
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")
    result = complete_chat([{"role": "user", "content": "hi"}], policy_path=path)
    assert result.text == "from-local"
    assert result.backend == "deepseek-direct-flash"
    assert len(requests) == 1
    assert requests[0][1] == "Bearer fake-deepseek-key"


def test_local_server_trickles_until_primary_share_then_fallback(tmp_path, monkeypatch, local_http):
    primary_closed = threading.Event()

    def respond(handler):
        if handler.headers["Authorization"] == "Bearer fake-deepseek-key":
            handler.send_response(200)
            handler.end_headers()
            try:
                while True:
                    handler.wfile.write(b"\n")
                    handler.wfile.flush()
                    time.sleep(0.02)
            finally:
                primary_closed.set()
        else:
            send_local_answer(handler, "local-fallback", "z-ai/glm-5.3-flash")

    requests = local_http(respond)
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")
    started = time.monotonic()
    result = complete_chat([{"role": "user", "content": "hi"}], deadline_seconds=6, policy_path=path)
    elapsed = time.monotonic() - started
    assert result.text == "local-fallback"
    assert result.backend == "glm-openrouter-flash"
    assert 6 * 0.6 <= elapsed < 6
    assert len(requests) == 2
    assert primary_closed.wait(1)
    assert spend_state(tmp_path)["spent_usd"] > 0


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_local_server_redirect_never_forwards_a_key(tmp_path, monkeypatch, local_http, status):
    def respond(handler):
        if handler.headers["Authorization"] == "Bearer fake-deepseek-key":
            handler.send_response(status)
            handler.send_header("Location", "https://openrouter.ai/stolen-key")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
        else:
            send_local_answer(handler, "safe-fallback", "z-ai/glm-5.3-flash")

    requests = local_http(respond)
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")
    result = complete_chat([{"role": "user", "content": "hi"}], policy_path=path)
    assert result.text == "safe-fallback"
    assert [key for _, key in requests] == ["Bearer fake-deepseek-key", "Bearer fake-openrouter-key"]


def test_real_provider_hosts_are_blocked_by_suite_network_guard(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")

    # The test-only socket guard intentionally bubbles: application transport
    # code must not special-case RuntimeError from an embedding test harness.
    with pytest.raises(RuntimeError, match="network blocked in tests"):
        complete_chat([{"role": "user", "content": "must remain local"}], policy_path=path)


@pytest.mark.parametrize("host", ["api.deepseek.com", "openrouter.ai", "192.0.2.1", "2001:db8::1"])
def test_both_socket_entry_points_block_non_loopback(host):
    with pytest.raises(RuntimeError, match="^network blocked in tests$"):
        socket.create_connection((host, 443))
    # Reject before touching a socket; AF_INET construction is itself sandboxed.
    with pytest.raises(RuntimeError, match="^network blocked in tests$"):
        socket.socket.connect(None, (host, 443))


@pytest.mark.parametrize("failure", [
    b"not-json", b"[]", b'{"choices":[]}',
    b'{"choices":[{"message":{"content":""}}]}',
    ConnectionResetError("PRIVATE_PROVIDER_BODY"),
    http.client.IncompleteRead(b"PRIVATE_PROVIDER_BODY"),
])
def test_post_send_failures_charge_unknown_and_fall_back(tmp_path, monkeypatch, failure, caplog):
    class BrokenResponse(FakeResponse):
        def read1(self, size):
            if isinstance(failure, BaseException):
                raise failure
            return super().read1(size)

    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")
    broken = BrokenResponse(failure if isinstance(failure, bytes) else b"")
    opener = FakeOpener([broken, response("fallback", model="z-ai/glm-5.3-flash")])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)
    assert complete_chat([{"role": "user", "content": "hi"}], policy_path=path).text == "fallback"
    state = spend_state(tmp_path)
    assert state["spent_usd"] > 0
    assert state["reservations"] == {}
    assert broken.closed
    assert "PRIVATE_PROVIDER_BODY" not in json.dumps(state) + caplog.text


def test_unknown_settlement_at_cap_alerts_once_before_further_network(tmp_path, monkeypatch):
    def adjust(policy):
        lane = policy["app_lanes"]["chat_fast"]
        lane["daily_cap_usd"] = 0.000003
        lane["max_output_tokens"] = 1
        card = policy["app_backends"]["deepseek-direct-flash"]["price_card"]
        card.update(input_usd_per_million=1, output_usd_per_million=1)

    path = policy_file(tmp_path, mutate=adjust)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    class TimedOutResponse(FakeResponse):
        def read1(self, _size):
            raise TimeoutError("PRIVATE_PROVIDER_BODY")

    opener = FakeOpener([TimedOutResponse()])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)
    with pytest.raises(RouteUnavailable):
        complete_chat([{"role": "user", "content": "hello"}], policy_path=path)
    for _ in range(2):
        runtime_policy._RUNTIMES.pop(path, None)
        with pytest.raises(BudgetExhausted):
            complete_chat([{"role": "user", "content": "hello"}], policy_path=path)
    assert len(opener.requests) == 1
    alerts = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert [event["reason"] for event in alerts] == ["budget_80", "budget_100"]


@pytest.mark.parametrize("reply_model", [
    "deepseek-flash", "deepseek/deepseek-flash", "deepseek-flash:latest", "deepseek-flash-20260914",
])
def test_reply_model_variants_return_without_mismatch_alert(tmp_path, monkeypatch, reply_model):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    opener = FakeOpener([response(model=reply_model)])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)
    result = complete_chat([{"role": "user", "content": "hi"}], policy_path=path)
    assert result.text == "answer"
    assert result.reply_model == reply_model
    assert not (tmp_path / "alerts.jsonl").exists()


def test_fallback_is_skipped_with_less_than_two_seconds_remaining(tmp_path, monkeypatch):
    path = policy_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")
    opener = FakeOpener([FakeResponse(status=500)])
    monkeypatch.setattr(app_chat.urllib.request, "build_opener", lambda *_args: opener)
    with pytest.raises(RouteUnavailable):
        complete_chat([{"role": "user", "content": "hi"}], deadline_seconds=1.9, policy_path=path)
    assert len(opener.requests) == 1
