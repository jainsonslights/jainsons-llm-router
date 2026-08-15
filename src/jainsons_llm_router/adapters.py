"""Provider adapter boundary and built-in adapters.

Adapters normalize provider I/O. They never select fallbacks and never make a
budget decision; those are exclusively Router and UsageLedger responsibilities.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol

from .errors import InvalidRequest
from .models import (
    AdapterRequest,
    AdapterResult,
    ChargeEstimate,
    FailurePhase,
    LLMRequest,
    Usage,
)


class AdapterFailure(Exception):
    """A normalized provider failure with an explicit dispatch boundary."""

    def __init__(
        self,
        reason_code: str,
        *,
        phase: FailurePhase,
        actual_micro_usd: int | None = None,
        provider_request_id: str | None = None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.phase = phase
        self.actual_micro_usd = actual_micro_usd
        self.provider_request_id = provider_request_id


class ProviderAdapter(Protocol):
    """Small, provider-neutral adapter contract."""

    def validate_capability(self, request: LLMRequest, *, model: str, timeout_seconds: float) -> None: ...

    def estimate_max_charge(
        self, request: LLMRequest, *, model: str, price_card_version: str
    ) -> ChargeEstimate: ...

    def complete(self, request: AdapterRequest) -> AdapterResult: ...

    async def acomplete(self, request: AdapterRequest) -> AdapterResult: ...

    def is_non_billable(self, *, model: str, price_card_version: str | None) -> bool: ...

    def health(self) -> tuple[bool, str | None]: ...


@dataclass(frozen=True)
class PriceCard:
    """Integer prices per one million tokens plus fixed request charges."""

    input_micro_usd_per_million: int
    output_micro_usd_per_million: int
    fixed_micro_usd: int = 0
    rounding_margin_micro_usd: int = 0

    def __post_init__(self) -> None:
        values = (
            self.input_micro_usd_per_million,
            self.output_micro_usd_per_million,
            self.fixed_micro_usd,
            self.rounding_margin_micro_usd,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("price card amounts must be non-negative integer micro-USD")

    def charge(self, input_tokens: int, output_tokens: int, *, include_margin: bool) -> int:
        input_charge = (input_tokens * self.input_micro_usd_per_million + 999_999) // 1_000_000
        output_charge = (output_tokens * self.output_micro_usd_per_million + 999_999) // 1_000_000
        margin = self.rounding_margin_micro_usd if include_margin else 0
        return self.fixed_micro_usd + input_charge + output_charge + margin


class GenericHTTPAdapter:
    """A working OpenAI-shaped JSON-over-HTTP adapter.

    It uses :mod:`urllib` so the core package has no runtime dependency. The
    configured endpoint and secret environment-variable name belong to trusted
    deployment configuration. Raw headers and payloads are never exposed to the
    router's structured logs.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key_env: str,
        price_cards: Mapping[str, PriceCard],
        supported_models: Iterable[str],
        supported_modalities: Iterable[str] = ("text",),
        verified_free_models: Mapping[str, Iterable[str]] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        idempotency_header: str | None = "Idempotency-Key",
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if not endpoint.startswith(("https://", "http://")):
            raise ValueError("HTTP adapter endpoint must be http(s)")
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.price_cards = dict(price_cards)
        self.supported_models = frozenset(supported_models)
        self.supported_modalities = frozenset(supported_modalities)
        self.verified_free_models = {
            version: frozenset(models) for version, models in (verified_free_models or {}).items()
        }
        self.extra_headers = dict(extra_headers or {})
        self.idempotency_header = idempotency_header
        self._opener = opener or urllib.request.urlopen

    @staticmethod
    def _estimated_input_tokens(request: LLMRequest) -> int:
        # One token per three characters is deliberately conservative for a
        # provider-independent bound. Adapters may override this implementation.
        return max(1, (request.input_size + 2) // 3)

    def validate_capability(self, request: LLMRequest, *, model: str, timeout_seconds: float) -> None:
        if model not in self.supported_models:
            raise AdapterFailure("unsupported_model", phase=FailurePhase.PRE_DISPATCH)
        if request.modality not in self.supported_modalities:
            raise AdapterFailure("unsupported_modality", phase=FailurePhase.PRE_DISPATCH)
        if timeout_seconds <= 0 or request.timeout_seconds <= 0:
            raise AdapterFailure("invalid_timeout", phase=FailurePhase.PRE_DISPATCH)

    def estimate_max_charge(
        self, request: LLMRequest, *, model: str, price_card_version: str
    ) -> ChargeEstimate:
        card = self.price_cards.get(price_card_version)
        if card is None:
            raise AdapterFailure("unknown_price_card", phase=FailurePhase.PRE_DISPATCH)
        input_tokens = self._estimated_input_tokens(request)
        return ChargeEstimate(
            micro_usd=card.charge(input_tokens, request.max_output_tokens, include_margin=True),
            input_tokens=input_tokens,
        )

    def is_non_billable(self, *, model: str, price_card_version: str | None) -> bool:
        if price_card_version is None:
            return False
        return model in self.verified_free_models.get(price_card_version, ())

    def health(self) -> tuple[bool, str | None]:
        if not os.environ.get(self.api_key_env):
            return False, "credential_unavailable"
        return True, None

    def _payload(self, request: AdapterRequest) -> bytes:
        source = request.request
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": source.max_output_tokens,
            **dict(source.generation),
        }
        if source.messages is not None:
            payload["messages"] = [dict(item) for item in source.messages]
        else:
            payload["input"] = source.input
        if source.response_format is not None:
            payload["response_format"] = source.response_format
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def _parse(self, raw: bytes, *, price_card: PriceCard) -> AdapterResult:
        try:
            data = json.loads(raw.decode("utf-8"))
            usage_data = data.get("usage") or {}
            input_tokens = int(usage_data.get("input_tokens", usage_data.get("prompt_tokens", 0)) or 0)
            output_tokens = int(usage_data.get("output_tokens", usage_data.get("completion_tokens", 0)) or 0)
            usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
            output = data.get("output")
            if output is None:
                choices = data.get("choices") or []
                if choices:
                    output = (choices[0].get("message") or {}).get("content", choices[0].get("text"))
            if output is None:
                raise ValueError("missing output")
            request_id = data.get("id") or data.get("request_id")
            charge = price_card.charge(input_tokens, output_tokens, include_margin=False)
            return AdapterResult(
                output=output,
                usage=usage,
                provider_request_id=str(request_id) if request_id is not None else None,
                actual_micro_usd=charge,
            )
        except Exception as exc:
            raise AdapterFailure("invalid_provider_response", phase=FailurePhase.UNKNOWN) from exc

    def complete(self, request: AdapterRequest) -> AdapterResult:
        secret = os.environ.get(self.api_key_env)
        if not secret:
            raise AdapterFailure("credential_unavailable", phase=FailurePhase.PRE_DISPATCH)
        try:
            body = self._payload(request)
            headers = {"Content-Type": "application/json", **self.extra_headers}
            headers["Authorization"] = f"Bearer {secret}"
            if request.idempotency_key and self.idempotency_header:
                headers[self.idempotency_header] = request.idempotency_key
            http_request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        except Exception as exc:
            raise AdapterFailure("request_build_failed", phase=FailurePhase.PRE_DISPATCH) from exc

        try:
            with self._opener(http_request, timeout=request.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            request_id = exc.headers.get("x-request-id") if exc.headers else None
            raise AdapterFailure(
                "provider_rejected", phase=FailurePhase.DEFINITIVE, actual_micro_usd=0,
                provider_request_id=request_id,
            ) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise AdapterFailure("provider_outcome_unknown", phase=FailurePhase.UNKNOWN) from exc
        except Exception as exc:
            raise AdapterFailure("provider_outcome_unknown", phase=FailurePhase.UNKNOWN) from exc

        card = self.price_cards.get(request.price_card_version or "")
        if card is None:
            raise AdapterFailure("unknown_price_card", phase=FailurePhase.UNKNOWN)
        return self._parse(raw, price_card=card)

    async def acomplete(self, request: AdapterRequest) -> AdapterResult:
        return await asyncio.to_thread(self.complete, request)


class NullAdapter:
    """An always-unavailable adapter for disabled or unprovisioned slots."""

    def validate_capability(self, request: LLMRequest, *, model: str, timeout_seconds: float) -> None:
        raise AdapterFailure("adapter_unavailable", phase=FailurePhase.PRE_DISPATCH)

    def estimate_max_charge(
        self, request: LLMRequest, *, model: str, price_card_version: str
    ) -> ChargeEstimate:
        raise AdapterFailure("adapter_unavailable", phase=FailurePhase.PRE_DISPATCH)

    def complete(self, request: AdapterRequest) -> AdapterResult:
        raise AdapterFailure("adapter_unavailable", phase=FailurePhase.PRE_DISPATCH)

    async def acomplete(self, request: AdapterRequest) -> AdapterResult:
        return self.complete(request)

    def is_non_billable(self, *, model: str, price_card_version: str | None) -> bool:
        return False

    def health(self) -> tuple[bool, str | None]:
        return False, "adapter_unavailable"


class FakeAdapter:
    """Deterministic adapter for tests; it never performs network I/O."""

    def __init__(
        self,
        outcomes: Iterable[AdapterResult | AdapterFailure | Callable[[AdapterRequest], AdapterResult]] = (),
        *,
        estimate_micro_usd: int = 100,
        nonbillable_models: Iterable[str] | None = None,
        healthy: bool = True,
    ) -> None:
        self.outcomes = deque(outcomes)
        self.estimate_micro_usd = estimate_micro_usd
        self.nonbillable_models = None if nonbillable_models is None else frozenset(nonbillable_models)
        self.healthy = healthy
        self.calls: list[AdapterRequest] = []
        self._lock = threading.Lock()

    def validate_capability(self, request: LLMRequest, *, model: str, timeout_seconds: float) -> None:
        if not self.healthy:
            raise AdapterFailure("adapter_unhealthy", phase=FailurePhase.PRE_DISPATCH)

    def estimate_max_charge(
        self, request: LLMRequest, *, model: str, price_card_version: str
    ) -> ChargeEstimate:
        if not price_card_version:
            raise AdapterFailure("unknown_price_card", phase=FailurePhase.PRE_DISPATCH)
        return ChargeEstimate(micro_usd=self.estimate_micro_usd, input_tokens=max(1, request.input_size // 3))

    def complete(self, request: AdapterRequest) -> AdapterResult:
        with self._lock:
            self.calls.append(request)
            outcome = self.outcomes.popleft() if self.outcomes else AdapterResult(
                output="fake-output",
                usage=Usage(input_tokens=1, output_tokens=1),
                provider_request_id=f"fake-{len(self.calls)}",
                actual_micro_usd=self.estimate_micro_usd,
            )
        if isinstance(outcome, AdapterFailure):
            raise outcome
        if callable(outcome):
            return outcome(request)
        return outcome

    async def acomplete(self, request: AdapterRequest) -> AdapterResult:
        return self.complete(request)

    def is_non_billable(self, *, model: str, price_card_version: str | None) -> bool:
        return self.nonbillable_models is None or model in self.nonbillable_models

    def health(self) -> tuple[bool, str | None]:
        return (True, None) if self.healthy else (False, "adapter_unhealthy")


class StubProviderAdapter(NullAdapter):
    """SDK integration placeholder that still obeys the complete contract."""

    provider_name = "stub"

    def __init__(self, *, reason_code: str = "sdk_adapter_not_configured") -> None:
        self.reason_code = reason_code

    def _raise(self) -> None:
        raise AdapterFailure(self.reason_code, phase=FailurePhase.PRE_DISPATCH)

    def validate_capability(self, request: LLMRequest, *, model: str, timeout_seconds: float) -> None:
        self._raise()

    def estimate_max_charge(
        self, request: LLMRequest, *, model: str, price_card_version: str
    ) -> ChargeEstimate:
        self._raise()
        raise AssertionError("unreachable")

    def complete(self, request: AdapterRequest) -> AdapterResult:
        self._raise()
        raise AssertionError("unreachable")

    def health(self) -> tuple[bool, str | None]:
        return False, self.reason_code


class AnthropicAdapter(StubProviderAdapter):
    provider_name = "anthropic"


class GeminiAdapter(StubProviderAdapter):
    provider_name = "gemini"


class OpenRouterAdapter(StubProviderAdapter):
    provider_name = "openrouter"


class MistralAdapter(StubProviderAdapter):
    provider_name = "mistral"
