from __future__ import annotations

import json
import os

import pytest

from jainsons_llm_router import (
    AdapterRequest,
    AnthropicAdapter,
    Candidate,
    FakeAdapter,
    GenericHTTPAdapter,
    InMemoryEventLogger,
    LLMRequest,
    PriceCard,
    Usage,
)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.body


def test_structured_logger_drops_prompts_completions_credentials_and_phone_numbers():
    prompt = "PROMPT-DO-NOT-LOG"
    completion = "COMPLETION-DO-NOT-LOG"
    api_key = "sk-test-secret-123456789"
    phone = "+91 98765 43210"
    logger = InMemoryEventLogger()
    logger.emit(
        {
            "prompt": prompt,
            "completion": completion,
            "api_key": api_key,
            "authorization": f"Bearer {api_key}",
            "headers": {"Authorization": f"Bearer {api_key}"},
            "reason_code": f"{prompt} {completion} {phone} Bearer {api_key}",
            "provider": "test-provider",
        }
    )
    rendered = json.dumps(logger.events)
    for forbidden in (prompt, completion, api_key, phone):
        assert forbidden not in rendered
    assert logger.events[0]["provider"] == "test-provider"


def test_generic_http_adapter_is_in_process_and_normalizes_response(monkeypatch):
    observed = {}

    def opener(request, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        observed["headers"] = dict(request.headers)
        observed["body"] = request.data
        return FakeResponse(
            b'{"id":"provider-request","output":"answer","usage":{"input_tokens":2,"output_tokens":3}}'
        )

    monkeypatch.setenv("TEST_ROUTER_KEY", "sk-test-secret-123456789")
    adapter = GenericHTTPAdapter(
        endpoint="https://provider.example/v1/generate",
        api_key_env="TEST_ROUTER_KEY",
        supported_models={"model"},
        price_cards={"v1": PriceCard(1_000_000, 2_000_000, fixed_micro_usd=7)},
        opener=opener,
    )
    request = LLMRequest(input="hello", max_output_tokens=9, idempotency_key="stable-key")
    result = adapter.complete(
        AdapterRequest(
            request=request,
            provider="provider",
            model="model",
            provider_account_alias="account",
            timeout_seconds=4.0,
            idempotency_key="stable-key",
            price_card_version="v1",
        )
    )
    assert result.output == "answer"
    assert result.usage == Usage(input_tokens=2, output_tokens=3)
    assert result.actual_micro_usd == 15
    assert observed["url"] == "https://provider.example/v1/generate"
    assert observed["timeout"] == 4.0
    assert b'"model":"model"' in observed["body"]
    assert observed["headers"]["Idempotency-key"] == "stable-key"


def test_provider_stubs_implement_adapter_contract_without_subprocesses():
    adapter = AnthropicAdapter()
    request = LLMRequest(input="hello")
    with pytest.raises(Exception, match="sdk_adapter_not_configured"):
        adapter.validate_capability(request, model="model", timeout_seconds=1)
    assert adapter.health() == (False, "sdk_adapter_not_configured")
