"""Generated from harness.py routing export. DO NOT EDIT BY HAND.

Regenerate with ``python -m jainsons_llm_router.sync_from_harness``.
The digest covers models, HTTP chains, and the harness free-check hash.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .. import free_check
from ..errors import ConfigurationError
from ..models import BillingClass, Candidate


@dataclass(frozen=True)
class HarnessBackendPolicy:
    name: str
    kind: str
    funding: str
    billing_class: BillingClass
    model: str | None
    model_source: str
    automatic_enabled: bool


HARNESS_POLICY_SHA256 = '8cf4bb94ea95479f8fa4217de804872ed161afa16e0adb8dbf6ab7bdb19656ce'
HARNESS_FREE_CHECK_SHA256 = '012725adf46246795134fe835f9be6c4ebefd6304f3c276003265fd7cc37186b'
DEFAULT_LANE = 'research'
AUTO_DISABLED_BACKENDS = frozenset(('claude', 'claude-opus', 'claude-sonnet', 'glm', 'kimi', 'or-best'))
GLM_AUTOMATIC_DISABLED = 'glm' in AUTO_DISABLED_BACKENDS

BACKENDS = MappingProxyType({
    'agy': HarnessBackendPolicy(
        name='agy', kind='sub-free', funding='subscription',
        billing_class=BillingClass.FREE, model=None,
        model_source='harness routing export', automatic_enabled=True,
    ),
    'claude': HarnessBackendPolicy(
        name='claude', kind='sub-anthropic', funding='subscription',
        billing_class=BillingClass.FREE, model=None,
        model_source='harness routing export', automatic_enabled=False,
    ),
    'claude-opus': HarnessBackendPolicy(
        name='claude-opus', kind='sub-anthropic', funding='subscription',
        billing_class=BillingClass.FREE, model='claude-opus-5-5',
        model_source='harness routing export', automatic_enabled=False,
    ),
    'claude-sonnet': HarnessBackendPolicy(
        name='claude-sonnet', kind='sub-anthropic', funding='subscription',
        billing_class=BillingClass.FREE, model='claude-sonnet-5',
        model_source='harness routing export', automatic_enabled=False,
    ),
    'codex': HarnessBackendPolicy(
        name='codex', kind='sub', funding='subscription',
        billing_class=BillingClass.FREE, model='gpt-5.6-terra',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'codex-astra': HarnessBackendPolicy(
        name='codex-astra', kind='sub', funding='subscription',
        billing_class=BillingClass.FREE, model='gpt-6-astra',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'codex-luna': HarnessBackendPolicy(
        name='codex-luna', kind='sub', funding='subscription',
        billing_class=BillingClass.FREE, model='gpt-6-luna',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'codex-sol': HarnessBackendPolicy(
        name='codex-sol', kind='sub', funding='subscription',
        billing_class=BillingClass.FREE, model='gpt-6-sol',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'glm': HarnessBackendPolicy(
        name='glm', kind='sub-glm', funding='subscription',
        billing_class=BillingClass.FREE, model=None,
        model_source='harness routing export', automatic_enabled=False,
    ),
    'kimi': HarnessBackendPolicy(
        name='kimi', kind='sub-kimi', funding='subscription',
        billing_class=BillingClass.FREE, model=None,
        model_source='harness routing export', automatic_enabled=False,
    ),
    'omni-audit': HarnessBackendPolicy(
        name='omni-audit', kind='omni-free', funding='free_service',
        billing_class=BillingClass.FREE, model='omni-kiro-sonnet45',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'omni-diverse': HarnessBackendPolicy(
        name='omni-diverse', kind='omni-free', funding='free_service',
        billing_class=BillingClass.FREE, model='omni-kiro-deepseek',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'omni-fast': HarnessBackendPolicy(
        name='omni-fast', kind='omni-free', funding='free_service',
        billing_class=BillingClass.FREE, model='omni-groq-llama',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'or-best': HarnessBackendPolicy(
        name='or-best', kind='API$$', funding='paid_api',
        billing_class=BillingClass.PAID, model=None,
        model_source='harness routing export', automatic_enabled=False,
    ),
    'or-free-gemma4': HarnessBackendPolicy(
        name='or-free-gemma4', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model='google/gemma-4-31b-it:free',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'or-free-laguna-s': HarnessBackendPolicy(
        name='or-free-laguna-s', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model='poolside/laguna-s-2.1:free',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'or-free-laguna-xs': HarnessBackendPolicy(
        name='or-free-laguna-xs', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model='poolside/laguna-xs-2.1:free',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'or-free-ling': HarnessBackendPolicy(
        name='or-free-ling', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model='inclusionai/ling-3.0-flash:free',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'or-free-nemotron': HarnessBackendPolicy(
        name='or-free-nemotron', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model='nvidia/nemotron-3-ultra-550b-a55b:free',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'or-free-north-code': HarnessBackendPolicy(
        name='or-free-north-code', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model='cohere/north-mini-code:free',
        model_source='harness routing export', automatic_enabled=True,
    ),
})

LANE_BACKEND_ORDER = MappingProxyType({
    'code': ('codex', 'kimi'),
    'domain_ops': ('agy', 'codex'),
    'image_gen': ('agy', 'agy'),
    'legal_finance': ('claude', 'codex'),
    'planning': ('claude', 'codex'),
    'research': ('agy', 'codex'),
    'ui': ('kimi', 'codex'),
    'vision': ('agy', 'agy'),
    'writing': ('codex', 'agy'),
})

FREE_CANDIDATE_BACKENDS_BY_LANE = MappingProxyType({
    'code': ('or-free-ling',),
    'domain_ops': (),
    'image_gen': (),
    'legal_finance': (),
    'planning': (),
    'research': ('or-free-nemotron', 'or-free-laguna-s', 'or-free-laguna-xs', 'or-free-north-code', 'or-free-ling', 'or-free-gemma4'),
    'ui': ('or-free-ling',),
    'vision': (),
    'writing': ('or-free-nemotron', 'or-free-laguna-s', 'or-free-laguna-xs', 'or-free-north-code', 'or-free-ling', 'or-free-gemma4'),
})

HTTP_CHAIN_BY_LANE = MappingProxyType({
    'code': ('or-free-ling',),
    'domain_ops': (),
    'image_gen': (),
    'legal_finance': (),
    'planning': (),
    'research': ('or-free-nemotron', 'or-free-laguna-s', 'or-free-laguna-xs', 'or-free-north-code', 'or-free-ling', 'or-free-gemma4'),
    'ui': ('or-free-ling',),
    'vision': (),
    'writing': ('or-free-nemotron', 'or-free-laguna-s', 'or-free-laguna-xs', 'or-free-north-code', 'or-free-ling', 'or-free-gemma4'),
})

OPENROUTER_FREE_BACKENDS = ('or-free-nemotron', 'or-free-laguna-s', 'or-free-laguna-xs', 'or-free-north-code', 'or-free-ling', 'or-free-gemma4')

def free_candidate_backend_order(lane: str) -> tuple[str, ...]:
    try:
        configured = FREE_CANDIDATE_BACKENDS_BY_LANE[lane]
    except KeyError as exc:
        raise ConfigurationError(f'unknown harness lane: {lane}') from exc
    return tuple(
        name for name in configured
        if (policy := BACKENDS.get(name)) is not None
        and policy.kind == 'or-free'
        and policy.model is not None
        and free_check.model_is_free(policy.model)
    )


def order_free_candidates(lane: str, candidates_by_backend: Mapping[str, Candidate], *, allow_missing: Iterable[str] = ()) -> tuple[Candidate, ...]:
    order = free_candidate_backend_order(lane)
    allowed = frozenset(allow_missing)
    unknown_allowed = allowed.difference(order)
    if unknown_allowed:
        raise ConfigurationError(f'allow_missing contains backends outside lane {lane}: {sorted(unknown_allowed)}')
    missing = [name for name in order if name not in candidates_by_backend and name not in allowed]
    if missing:
        raise ConfigurationError(f'missing harness-derived free candidates for lane {lane}: {missing}')
    selected: list[Candidate] = []
    for name in order:
        policy = BACKENDS[name]
        candidate = candidates_by_backend.get(name)
        if candidate is None:
            continue
        if candidate.billing_class is not BillingClass.FREE or not candidate.zero_marginal_cost:
            raise ConfigurationError(f'harness backend {name} must map to a zero-cost free Candidate')
        if candidate.model != policy.model or not free_check.model_is_free(policy.model):
            continue
        selected.append(candidate)
    return tuple(selected)
