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


HARNESS_POLICY_SHA256 = '0926959cc5a1f21b42153d5155f24b137838af151372f078e4a7e5949f7675e0'
HARNESS_FREE_CHECK_SHA256 = 'f7acf1713e37a0139eff17c9f04b51036c8af2400cbfb315bc02960f4e1e7b73'
DEFAULT_LANE = 'research'
AUTO_DISABLED_BACKENDS = frozenset(('claude', 'claude-opus', 'claude-sonnet', 'codex-astra', 'codex-terra', 'glm', 'kimi', 'omni-diverse', 'omni-fast', 'or-best', 'or-free-gemma4', 'or-free-ling'))
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
        model_source='harness routing export', automatic_enabled=False,
    ),
    'codex-luna': HarnessBackendPolicy(
        name='codex-luna', kind='sub', funding='subscription',
        billing_class=BillingClass.FREE, model='gpt-6-luna',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'codex-sol': HarnessBackendPolicy(
        name='codex-sol', kind='sub', funding='subscription',
        billing_class=BillingClass.FREE, model='gpt-6.1-sol',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'codex-sol-web': HarnessBackendPolicy(
        name='codex-sol-web', kind='sub', funding='subscription',
        billing_class=BillingClass.FREE, model='gpt-6.1-sol',
        model_source='harness routing export', automatic_enabled=True,
    ),
    'codex-terra': HarnessBackendPolicy(
        name='codex-terra', kind='sub', funding='subscription',
        billing_class=BillingClass.FREE, model='gpt-5.6-terra',
        model_source='harness routing export', automatic_enabled=False,
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
        model_source='harness routing export', automatic_enabled=False,
    ),
    'omni-fast': HarnessBackendPolicy(
        name='omni-fast', kind='omni-free', funding='free_service',
        billing_class=BillingClass.FREE, model='omni-groq-llama',
        model_source='harness routing export', automatic_enabled=False,
    ),
    'or-best': HarnessBackendPolicy(
        name='or-best', kind='API$$', funding='paid_api',
        billing_class=BillingClass.PAID, model=None,
        model_source='harness routing export', automatic_enabled=False,
    ),
    'or-free-gemma4': HarnessBackendPolicy(
        name='or-free-gemma4', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model='google/gemma-4-31b-it:free',
        model_source='harness routing export', automatic_enabled=False,
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
        model_source='harness routing export', automatic_enabled=False,
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
    'or-free-space-bunny': HarnessBackendPolicy(
        name='or-free-space-bunny', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model='stealth/space-bunny-alpha',
        model_source='harness routing export', automatic_enabled=True,
    ),
})

LANE_BACKEND_ORDER = MappingProxyType({
    'code': ('codex-sol', 'codex-luna'),
    'domain_ops': ('agy', 'codex-luna'),
    'image_gen': ('agy', 'agy'),
    'legal_finance': ('claude', 'codex-sol-web'),
    'planning': ('claude', 'codex-sol'),
    'research': ('codex-sol-web', 'agy'),
    'ui': ('codex-sol', 'codex-luna'),
    'vision': ('agy', 'agy'),
    'writing': ('codex-luna', 'agy'),
})

FREE_CANDIDATE_BACKENDS_BY_LANE = MappingProxyType({
    'code': (),
    'domain_ops': (),
    'image_gen': (),
    'legal_finance': (),
    'planning': (),
    'research': ('or-free-space-bunny', 'or-free-nemotron', 'or-free-laguna-s', 'or-free-laguna-xs', 'or-free-north-code'),
    'ui': (),
    'vision': (),
    'writing': ('or-free-space-bunny', 'or-free-nemotron', 'or-free-laguna-s', 'or-free-laguna-xs', 'or-free-north-code'),
})

HTTP_CHAIN_BY_LANE = MappingProxyType({
    'code': (),
    'domain_ops': (),
    'image_gen': (),
    'legal_finance': (),
    'planning': (),
    'research': ('or-free-space-bunny', 'or-free-nemotron', 'or-free-laguna-s', 'or-free-laguna-xs', 'or-free-north-code'),
    'ui': (),
    'vision': (),
    'writing': ('or-free-space-bunny', 'or-free-nemotron', 'or-free-laguna-s', 'or-free-laguna-xs', 'or-free-north-code'),
})

OPENROUTER_FREE_BACKENDS = ('or-free-space-bunny', 'or-free-nemotron', 'or-free-laguna-s', 'or-free-laguna-xs', 'or-free-north-code', 'or-free-ling', 'or-free-gemma4')

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
