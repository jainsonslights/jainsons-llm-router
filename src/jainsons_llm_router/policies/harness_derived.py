"""Generated from harness.py static routing policy. DO NOT EDIT BY HAND.

Regenerate with ``python -m jainsons_llm_router.sync_from_harness``.
The digest covers extracted policy facts only, not harness runtime logic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

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


HARNESS_POLICY_SHA256 = '2e2b479c353902f6482d6bc07fb5cd5edc1e26f8fb7c30130e458e51a43c5bfb'
DEFAULT_LANE = 'research'
AUTO_DISABLED_BACKENDS = frozenset(('glm', 'kimi'))
GLM_AUTOMATIC_DISABLED = 'glm' in AUTO_DISABLED_BACKENDS

BACKENDS = MappingProxyType({
    'agy': HarnessBackendPolicy(
        name='agy', kind='sub-free', funding='subscription',
        billing_class=BillingClass.FREE, model=None,
        model_source='provider-default', automatic_enabled=True,
    ),
    'agy-pro': HarnessBackendPolicy(
        name='agy-pro', kind='sub-free', funding='subscription',
        billing_class=BillingClass.FREE, model='gemini-3.1-pro-high',
        model_source="'gemini-3.1-pro-high'", automatic_enabled=True,
    ),
    'claude': HarnessBackendPolicy(
        name='claude', kind='sub-anthropic', funding='subscription',
        billing_class=BillingClass.FREE, model=None,
        model_source='provider-default', automatic_enabled=True,
    ),
    'claude-fable': HarnessBackendPolicy(
        name='claude-fable', kind='sub-anthropic', funding='subscription',
        billing_class=BillingClass.FREE, model='claude-fable-5',
        model_source="os.environ.get('HARNESS_FABLE_MODEL', 'claude-fable-5')", automatic_enabled=True,
    ),
    'claude-opus': HarnessBackendPolicy(
        name='claude-opus', kind='sub-anthropic', funding='subscription',
        billing_class=BillingClass.FREE, model='claude-opus-5',
        model_source="'claude-opus-5'", automatic_enabled=True,
    ),
    'codex': HarnessBackendPolicy(
        name='codex', kind='sub', funding='subscription',
        billing_class=BillingClass.FREE, model='gpt-5.6-terra',
        model_source='CODEX_MODEL', automatic_enabled=True,
    ),
    'codex-luna': HarnessBackendPolicy(
        name='codex-luna', kind='sub', funding='subscription',
        billing_class=BillingClass.FREE, model='gpt-5.6-luna',
        model_source="'gpt-5.6-luna'", automatic_enabled=True,
    ),
    'codex-sol': HarnessBackendPolicy(
        name='codex-sol', kind='sub', funding='subscription',
        billing_class=BillingClass.FREE, model='gpt-5.6-sol',
        model_source="'gpt-5.6-sol'", automatic_enabled=True,
    ),
    'glm': HarnessBackendPolicy(
        name='glm', kind='sub-glm', funding='subscription',
        billing_class=BillingClass.FREE, model='glm-5.2',
        model_source="BACKEND_ENV['glm'].ANTHROPIC_DEFAULT_SONNET_MODEL (GLM_MODEL)", automatic_enabled=False,
    ),
    'kimi': HarnessBackendPolicy(
        name='kimi', kind='sub-kimi', funding='subscription',
        billing_class=BillingClass.FREE, model='k3[1m]',
        model_source="BACKEND_ENV['kimi'].ANTHROPIC_DEFAULT_SONNET_MODEL (KIMI_MODEL)", automatic_enabled=False,
    ),
    'media_ocr': HarnessBackendPolicy(
        name='media_ocr', kind='media', funding='free_service',
        billing_class=BillingClass.FREE, model=None,
        model_source='provider-default', automatic_enabled=True,
    ),
    'omni-audit': HarnessBackendPolicy(
        name='omni-audit', kind='omni-free', funding='free_service',
        billing_class=BillingClass.FREE, model='omni-kiro-sonnet45',
        model_source="'omni-kiro-sonnet45'", automatic_enabled=True,
    ),
    'omni-diverse': HarnessBackendPolicy(
        name='omni-diverse', kind='omni-free', funding='free_service',
        billing_class=BillingClass.FREE, model='omni-kiro-deepseek',
        model_source="'omni-kiro-deepseek'", automatic_enabled=True,
    ),
    'omni-fast': HarnessBackendPolicy(
        name='omni-fast', kind='omni-free', funding='free_service',
        billing_class=BillingClass.FREE, model='omni-groq-llama',
        model_source="'omni-groq-llama'", automatic_enabled=True,
    ),
    'or-best': HarnessBackendPolicy(
        name='or-best', kind='API$$', funding='paid_api',
        billing_class=BillingClass.PAID, model=None,
        model_source='BEST_MODEL', automatic_enabled=False,
    ),
    'or-free-gemma4': HarnessBackendPolicy(
        name='or-free-gemma4', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model=None,
        model_source="CATALOG['or-free-gemma4']['id']", automatic_enabled=True,
    ),
    'or-free-laguna-s': HarnessBackendPolicy(
        name='or-free-laguna-s', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model=None,
        model_source="CATALOG['or-free-laguna-s']['id']", automatic_enabled=True,
    ),
    'or-free-laguna-xs': HarnessBackendPolicy(
        name='or-free-laguna-xs', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model=None,
        model_source="CATALOG['or-free-laguna-xs']['id']", automatic_enabled=True,
    ),
    'or-free-ling': HarnessBackendPolicy(
        name='or-free-ling', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model=None,
        model_source="CATALOG['or-free-ling']['id']", automatic_enabled=True,
    ),
    'or-free-nemotron': HarnessBackendPolicy(
        name='or-free-nemotron', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model=None,
        model_source="CATALOG['or-free-nemotron']['id']", automatic_enabled=True,
    ),
    'or-free-north-code': HarnessBackendPolicy(
        name='or-free-north-code', kind='or-free', funding='free_service',
        billing_class=BillingClass.FREE, model=None,
        model_source="CATALOG['or-free-north-code']['id']", automatic_enabled=True,
    ),
})

LANE_BACKEND_ORDER = MappingProxyType({
    'code': ('codex', 'agy'),
    'domain_ops': ('agy', 'codex'),
    'image_gen': ('agy', 'agy'),
    'legal_finance': ('claude', 'claude'),
    'ocr': ('media_ocr', 'media_ocr'),
    'planning': ('claude', 'codex'),
    'research': ('agy', 'codex'),
    'ui': ('codex', 'agy'),
    'vision': ('agy', 'agy'),
    'writing': ('codex', 'agy'),
})

FREE_CANDIDATE_BACKENDS_BY_LANE = MappingProxyType({
    'code': ('codex', 'agy'),
    'domain_ops': ('agy', 'codex'),
    'image_gen': ('agy',),
    'legal_finance': ('claude',),
    'ocr': ('media_ocr',),
    'planning': ('claude', 'codex'),
    'research': ('agy', 'codex'),
    'ui': ('codex', 'agy'),
    'vision': ('agy',),
    'writing': ('codex', 'agy'),
})


def free_candidate_backend_order(lane: str) -> tuple[str, ...]:
    try:
        return FREE_CANDIDATE_BACKENDS_BY_LANE[lane]
    except KeyError as exc:
        raise ConfigurationError(f'unknown harness lane: {lane}') from exc


def order_free_candidates(
    lane: str,
    candidates_by_backend: Mapping[str, Candidate],
    *,
    allow_missing: Iterable[str] = (),
) -> tuple[Candidate, ...]:
    """Apply harness order to deployment-owned free Candidate objects.

    Missing harness backends fail unless explicitly acknowledged in
    ``allow_missing``. Paid candidates, caps, approvals, adapter/account
    wiring, and exact-model pins remain local to the consuming route.
    """

    order = free_candidate_backend_order(lane)
    allowed = frozenset(allow_missing)
    unknown_allowed = allowed.difference(order)
    if unknown_allowed:
        raise ConfigurationError(
            f'allow_missing contains backends outside lane {lane}: {sorted(unknown_allowed)}'
        )
    missing = [name for name in order if name not in candidates_by_backend and name not in allowed]
    if missing:
        raise ConfigurationError(
            f'missing harness-derived free candidates for lane {lane}: {missing}'
        )
    selected: list[Candidate] = []
    for name in order:
        candidate = candidates_by_backend.get(name)
        if candidate is None:
            continue
        if candidate.billing_class is not BillingClass.FREE:
            raise ConfigurationError(f'harness backend {name} must map to a free Candidate')
        if not candidate.zero_marginal_cost:
            raise ConfigurationError(
                f'harness backend {name} must prove zero marginal cost'
            )
        selected.append(candidate)
    return tuple(selected)

