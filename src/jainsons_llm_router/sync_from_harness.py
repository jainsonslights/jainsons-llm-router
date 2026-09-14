"""Generate router routing policy from the harness's JSON export.

The harness is the routing-policy authority. This module deliberately runs
``harness.py --routing-json`` rather than parsing its implementation.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .free_check import PORTED_FROM_HARNESS_SHA256

DEFAULT_HARNESS_PATH = Path.home() / ".claude" / "skills" / "harness-offload" / "harness.py"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "policies" / "harness_derived.py"
_EXPORT_SCHEMA = 1


class HarnessSyncError(RuntimeError):
    """The harness routing export is missing, malformed, or unsafe."""


@dataclass(frozen=True)
class BackendFact:
    name: str
    kind: str
    funding: str
    billing_class: str
    model: str | None
    model_source: str
    automatic_enabled: bool
    http_openrouter: bool


@dataclass(frozen=True)
class HarnessPolicyFacts:
    backends: Mapping[str, BackendFact]
    lane_table: Mapping[str, tuple[str, ...]]
    escalation_chains: Mapping[str, tuple[str, ...]]
    http_chains: Mapping[str, tuple[str, ...]]
    openrouter_free_backends: tuple[str, ...]
    free_check_sha256: str
    default_lane: str

    @property
    def auto_disabled_backends(self) -> frozenset[str]:
        return frozenset(name for name, fact in self.backends.items() if not fact.automatic_enabled)

    @property
    def free_candidate_backends_by_lane(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}
        for lane, order in self.escalation_chains.items():
            selected: list[str] = []
            for name in order:
                fact = self.backends[name]
                if fact.kind == "or-free" and fact.automatic_enabled and name not in selected:
                    selected.append(name)
            result[lane] = tuple(selected)
        return result

    @property
    def policy_sha256(self) -> str:
        payload = {
            "backends": {name: {
                "automatic_enabled": fact.automatic_enabled,
                "billing_class": fact.billing_class,
                "funding": fact.funding,
                "http_openrouter": fact.http_openrouter,
                "kind": fact.kind,
                "model": fact.model,
            } for name, fact in sorted(self.backends.items())},
            "default_lane": self.default_lane,
            "escalation_chains": {lane: list(order) for lane, order in sorted(self.escalation_chains.items())},
            "free_check_sha256": self.free_check_sha256,
            "http_chains": {lane: list(order) for lane, order in sorted(self.http_chains.items())},
            "lane_table": {lane: list(order) for lane, order in sorted(self.lane_table.items())},
            "openrouter_free_backends": list(self.openrouter_free_backends),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise HarnessSyncError(f"routing export missing required {context}.{key}") from exc


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise HarnessSyncError(f"routing export {context} must be a non-empty string")
    return value


def _string_list(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise HarnessSyncError(f"routing export {context} must be a list of non-empty strings")
    return tuple(value)


def _classify_kind(kind: str) -> tuple[str, str]:
    if kind == "API$$":
        return "paid_api", "paid"
    if kind in {"sub", "sub-free", "sub-anthropic", "sub-glm", "sub-kimi"}:
        return "subscription", "free"
    if kind in {"or-free", "omni-free", "media"}:
        return "free_service", "free"
    raise HarnessSyncError(f"routing export backend has unknown kind {kind!r}")


def parse_routing_export(export: Any) -> HarnessPolicyFacts:
    """Validate and normalize one schema-1 harness routing export."""
    if not isinstance(export, dict):
        raise HarnessSyncError("routing export must be a JSON object")
    if export.get("schema") != _EXPORT_SCHEMA:
        raise HarnessSyncError(f"routing export schema mismatch: expected {_EXPORT_SCHEMA}, got {export.get('schema')!r}")

    default_lane = _string(_required(export, "default_lane", "root"), "default_lane")
    raw_backends = _required(export, "backends", "root")
    if not isinstance(raw_backends, dict) or not raw_backends:
        raise HarnessSyncError("routing export backends must be a non-empty object")
    backends: dict[str, BackendFact] = {}
    for name, raw in raw_backends.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise HarnessSyncError("routing export backends must map backend names to objects")
        kind = _string(_required(raw, "kind", f"backends[{name!r}]"), f"backends[{name!r}].kind")
        model = _required(raw, "model", f"backends[{name!r}]")
        if model == "":
            model = None  # harness V82 exports unset models (e.g. or-best without HARNESS_BEST_MODEL) as ""
        if model is not None and (not isinstance(model, str) or not model):
            raise HarnessSyncError(f"routing export backends[{name!r}].model must be a string or null")
        auto_disabled = _required(raw, "auto_disabled", f"backends[{name!r}]")
        http_openrouter = _required(raw, "http_openrouter", f"backends[{name!r}]")
        if not isinstance(auto_disabled, bool) or not isinstance(http_openrouter, bool):
            raise HarnessSyncError(f"routing export backends[{name!r}] boolean fields are invalid")
        if http_openrouter != (kind == "or-free"):
            raise HarnessSyncError(f"routing export backends[{name!r}].http_openrouter must be true only for kind 'or-free'")
        if kind == "or-free" and model is None:
            raise HarnessSyncError(f"routing export or-free backend {name!r} has null model")
        if kind == "or-free" and model.startswith("openrouter/"):
            raise HarnessSyncError(
                f"routing export or-free backend {name!r} has forbidden openrouter/ model prefix"
            )
        funding, billing_class = _classify_kind(kind)
        # The harness excludes capped Anthropic subscriptions and paid API
        # routes from automatic dispatch, in addition to export opt-outs.
        automatic = (not auto_disabled) and kind != "sub-anthropic" and funding != "paid_api"
        backends[name] = BackendFact(name, kind, funding, billing_class, model, "harness routing export", automatic, http_openrouter)

    openrouter_free_backends = _string_list(_required(export, "openrouter_free_backends", "root"), "openrouter_free_backends")
    if len(set(openrouter_free_backends)) != len(openrouter_free_backends):
        raise HarnessSyncError("routing export openrouter_free_backends contains duplicates")
    for name in openrouter_free_backends:
        fact = backends.get(name)
        if fact is None or fact.kind != "or-free" or fact.model is None:
            raise HarnessSyncError(f"routing export openrouter_free_backends has invalid backend {name!r}")

    raw_check = _required(export, "free_check", "root")
    if not isinstance(raw_check, dict):
        raise HarnessSyncError("routing export free_check must be an object")
    _string(_required(raw_check, "function", "free_check"), "free_check.function")
    free_check_sha256 = _string(_required(raw_check, "source_sha256", "free_check"), "free_check.source_sha256")

    raw_lanes = _required(export, "lanes", "root")
    if not isinstance(raw_lanes, dict) or not raw_lanes:
        raise HarnessSyncError("routing export lanes must be a non-empty object")
    lane_table: dict[str, tuple[str, ...]] = {}
    escalation_chains: dict[str, tuple[str, ...]] = {}
    http_chains: dict[str, tuple[str, ...]] = {}
    for lane, raw in raw_lanes.items():
        if not isinstance(lane, str) or not lane or not isinstance(raw, dict):
            raise HarnessSyncError("routing export lanes must map lane names to objects")
        primary = _string(_required(raw, "primary", f"lanes[{lane!r}]"), f"lanes[{lane!r}].primary")
        fallback = _string(_required(raw, "fallback", f"lanes[{lane!r}]"), f"lanes[{lane!r}].fallback")
        if primary not in backends or fallback not in backends:
            raise HarnessSyncError(f"routing export lanes[{lane!r}] references an unknown primary or fallback")
        escalation = _string_list(_required(raw, "escalation_chain", f"lanes[{lane!r}]"), f"lanes[{lane!r}].escalation_chain")
        http = _string_list(_required(raw, "http_chain", f"lanes[{lane!r}]"), f"lanes[{lane!r}].http_chain")
        for chain_name in (*escalation, *http):
            fact = backends.get(chain_name)
            if fact is None:
                raise HarnessSyncError(f"routing export lane {lane!r} references unknown backend {chain_name!r}")
            if chain_name.startswith("claude") or fact.billing_class == "paid":
                raise HarnessSyncError(f"routing export lane {lane!r} includes forbidden Claude/paid backend {chain_name!r}")
        for chain_name in http:
            if not backends[chain_name].http_openrouter:
                raise HarnessSyncError(f"routing export lane {lane!r} http_chain includes non-OpenRouter backend {chain_name!r}")
        lane_table[lane] = (primary, fallback)
        escalation_chains[lane] = escalation
        http_chains[lane] = http
    if default_lane not in lane_table:
        raise HarnessSyncError("routing export default_lane must name an exported lane")
    glm = backends.get("glm")
    if glm is not None and glm.automatic_enabled:
        raise HarnessSyncError("routing export enables GLM for automatic routing; human review required")
    return HarnessPolicyFacts(backends, lane_table, escalation_chains, http_chains, openrouter_free_backends, free_check_sha256, default_lane)


def read_harness_policy(path: str | Path) -> HarnessPolicyFacts:
    """Run the harness export in an empty temporary working directory."""
    try:
        with tempfile.TemporaryDirectory() as cwd:
            completed = subprocess.run(
                [sys.executable, str(Path(path).expanduser()), "--routing-json"],
                cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=90, check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise HarnessSyncError("harness routing export timed out after 90 seconds") from exc
    except OSError as exc:
        raise HarnessSyncError(f"cannot run harness routing export: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise HarnessSyncError(f"harness routing export exited {completed.returncode}{(': ' + detail) if detail else ''}")
    try:
        return parse_routing_export(json.loads(completed.stdout))
    except json.JSONDecodeError as exc:
        raise HarnessSyncError("harness routing export returned invalid JSON") from exc


def _quoted_tuple(values: Iterable[str]) -> str:
    items = tuple(values)
    if not items:
        return "()"
    return f"({items[0]!r},)" if len(items) == 1 else "(" + ", ".join(repr(item) for item in items) + ")"


def render_policy_module(facts: HarnessPolicyFacts) -> str:
    """Render deterministic, importable policy data from an export."""
    lines = [
        '"""Generated from harness.py routing export. DO NOT EDIT BY HAND.\n', "\n",
        "Regenerate with ``python -m jainsons_llm_router.sync_from_harness``.\n",
        "The digest covers models, HTTP chains, and the harness free-check hash.\n", '\"\"\"\n\n',
        "from __future__ import annotations\n\nfrom collections.abc import Iterable, Mapping\n",
        "from dataclasses import dataclass\nfrom types import MappingProxyType\n\n",
        "from .. import free_check\nfrom ..errors import ConfigurationError\nfrom ..models import BillingClass, Candidate\n\n\n",
        "@dataclass(frozen=True)\nclass HarnessBackendPolicy:\n",
        "    name: str\n    kind: str\n    funding: str\n    billing_class: BillingClass\n",
        "    model: str | None\n    model_source: str\n    automatic_enabled: bool\n\n\n",
        f"HARNESS_POLICY_SHA256 = {facts.policy_sha256!r}\n",
        f"HARNESS_FREE_CHECK_SHA256 = {facts.free_check_sha256!r}\n",
        f"DEFAULT_LANE = {facts.default_lane!r}\n",
        f"AUTO_DISABLED_BACKENDS = frozenset({_quoted_tuple(sorted(facts.auto_disabled_backends))})\n",
        "GLM_AUTOMATIC_DISABLED = 'glm' in AUTO_DISABLED_BACKENDS\n\nBACKENDS = MappingProxyType({\n",
    ]
    for name, fact in sorted(facts.backends.items()):
        billing = "BillingClass.FREE" if fact.billing_class == "free" else "BillingClass.PAID"
        lines.extend([f"    {name!r}: HarnessBackendPolicy(\n", f"        name={name!r}, kind={fact.kind!r}, funding={fact.funding!r},\n", f"        billing_class={billing}, model={fact.model!r},\n", f"        model_source={fact.model_source!r}, automatic_enabled={fact.automatic_enabled!r},\n", "    ),\n"])
    lines.extend(["})\n\nLANE_BACKEND_ORDER = MappingProxyType({\n"])
    for lane, order in sorted(facts.lane_table.items()): lines.append(f"    {lane!r}: {_quoted_tuple(order)},\n")
    lines.extend(["})\n\nFREE_CANDIDATE_BACKENDS_BY_LANE = MappingProxyType({\n"])
    for lane, order in sorted(facts.free_candidate_backends_by_lane.items()): lines.append(f"    {lane!r}: {_quoted_tuple(order)},\n")
    lines.extend(["})\n\nHTTP_CHAIN_BY_LANE = MappingProxyType({\n"])
    for lane, order in sorted(facts.http_chains.items()): lines.append(f"    {lane!r}: {_quoted_tuple(order)},\n")
    lines.extend(["})\n\n", f"OPENROUTER_FREE_BACKENDS = {_quoted_tuple(facts.openrouter_free_backends)}\n\n",
        "def free_candidate_backend_order(lane: str) -> tuple[str, ...]:\n    try:\n        configured = FREE_CANDIDATE_BACKENDS_BY_LANE[lane]\n    except KeyError as exc:\n        raise ConfigurationError(f'unknown harness lane: {lane}') from exc\n    return tuple(\n        name for name in configured\n        if (policy := BACKENDS.get(name)) is not None\n        and policy.kind == 'or-free'\n        and policy.model is not None\n        and free_check.model_is_free(policy.model)\n    )\n\n\n",
        "def order_free_candidates(lane: str, candidates_by_backend: Mapping[str, Candidate], *, allow_missing: Iterable[str] = ()) -> tuple[Candidate, ...]:\n",
        "    order = free_candidate_backend_order(lane)\n    allowed = frozenset(allow_missing)\n    unknown_allowed = allowed.difference(order)\n",
        "    if unknown_allowed:\n        raise ConfigurationError(f'allow_missing contains backends outside lane {lane}: {sorted(unknown_allowed)}')\n",
        "    missing = [name for name in order if name not in candidates_by_backend and name not in allowed]\n    if missing:\n        raise ConfigurationError(f'missing harness-derived free candidates for lane {lane}: {missing}')\n",
        "    selected: list[Candidate] = []\n    for name in order:\n        policy = BACKENDS[name]\n        candidate = candidates_by_backend.get(name)\n        if candidate is None:\n            continue\n        if candidate.billing_class is not BillingClass.FREE or not candidate.zero_marginal_cost:\n            raise ConfigurationError(f'harness backend {name} must map to a zero-cost free Candidate')\n        if candidate.model != policy.model or not free_check.model_is_free(policy.model):\n            continue\n        selected.append(candidate)\n    return tuple(selected)\n",
    ])
    return "".join(lines)


def _drift_diff(current: str, expected: str, output_path: Path) -> str:
    return "\n".join(list(difflib.unified_diff(current.splitlines(), expected.splitlines(), fromfile=str(output_path), tofile=f"{output_path} (regenerated)", lineterm=""))[:80])


def sync(*, harness_path: str | Path = DEFAULT_HARNESS_PATH, output_path: str | Path = DEFAULT_OUTPUT_PATH, check: bool = False, accept_free_check_hash: bool = False) -> int:
    facts = read_harness_policy(harness_path)
    if accept_free_check_hash:
        print(facts.free_check_sha256)
        return 0
    if facts.free_check_sha256 != PORTED_FROM_HARNESS_SHA256:
        raise HarnessSyncError("harness free-check changed; re-port router free_check.py")
    rendered = render_policy_module(facts)
    output = Path(output_path).expanduser()
    if check:
        try: current = output.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"harness-sync: drift: generated policy is missing: {output}", file=sys.stderr); return 1
        except OSError as exc: raise HarnessSyncError(f"cannot read generated policy {output}: {exc}") from exc
        if current != rendered:
            print("harness-sync: drift detected; run `python -m jainsons_llm_router.sync_from_harness`", file=sys.stderr)
            print(_drift_diff(current, rendered, output), file=sys.stderr); return 1
        print(f"harness-sync: current ({output})"); return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8"); temporary.replace(output)
    except OSError as exc: raise HarnessSyncError(f"cannot write generated policy {output}: {exc}") from exc
    print(f"harness-sync: wrote {output}"); return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--accept-free-check-hash", action="store_true", help="print a changed free-check hash without writing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try: return sync(harness_path=args.harness, output_path=args.output, check=args.check, accept_free_check_hash=args.accept_free_check_hash)
    except HarnessSyncError as exc:
        print(f"harness-sync: error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
