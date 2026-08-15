"""Generate the router's importable free-route order from ``harness.py``.

This module deliberately parses the harness source with :mod:`ast`; it never
imports or executes the live development harness.  The static-source contract
is intentionally narrow and explicit so a harness refactor fails loudly:

* ``BACKENDS = {...}`` contains ``name: (argv, edits_files, kind)`` entries.
* Conditional backends use ``BACKENDS["name"] = (argv, edits_files, kind)``.
* Claude-compatible conditional backends may pin their model through
  ``BACKEND_ENV["name"]["ANTHROPIC_DEFAULT_SONNET_MODEL"]``.
* ``API_BACKENDS``, ``LANE_TABLE``, ``DEFAULT_LANE``, and
  ``AUTO_DISABLED_BACKENDS`` are statically assigned collections.
* Model defaults may be literals or ``os.environ.get(NAME, DEFAULT)`` values,
  such as ``CODEX_MODEL``, ``GLM_MODEL``, and ``KIMI_MODEL``.

Worker health, authentication, admission, backoff, retry, and availability
functions are outside this contract and are neither parsed nor generated.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_HARNESS_PATH = Path.home() / ".claude" / "skills" / "harness-offload" / "harness.py"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "policies" / "harness_derived.py"


class HarnessSyncError(RuntimeError):
    """The harness static-policy contract is missing, ambiguous, or unsafe."""


@dataclass(frozen=True)
class BackendFact:
    """One statically declared harness backend."""

    name: str
    kind: str
    funding: str
    billing_class: str
    model: str | None
    model_source: str
    automatic_enabled: bool


@dataclass(frozen=True)
class HarnessPolicyFacts:
    """Policy-only facts extracted from harness source."""

    backends: Mapping[str, BackendFact]
    lane_table: Mapping[str, tuple[str, ...]]
    default_lane: str
    auto_disabled_backends: frozenset[str]

    @property
    def free_candidate_backends_by_lane(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}
        for lane, order in self.lane_table.items():
            selected: list[str] = []
            for name in order:
                backend = self.backends[name]
                if (
                    backend.billing_class == "free"
                    and backend.automatic_enabled
                    and name not in selected
                ):
                    selected.append(name)
            result[lane] = tuple(selected)
        return result

    @property
    def policy_sha256(self) -> str:
        payload = {
            "auto_disabled_backends": sorted(self.auto_disabled_backends),
            "backends": {
                name: {
                    "automatic_enabled": fact.automatic_enabled,
                    "billing_class": fact.billing_class,
                    "funding": fact.funding,
                    "kind": fact.kind,
                    "model": fact.model,
                    "model_source": fact.model_source,
                }
                for name, fact in sorted(self.backends.items())
            },
            "default_lane": self.default_lane,
            "lane_table": {name: list(order) for name, order in sorted(self.lane_table.items())},
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class _StaticEvaluator:
    """Evaluate only the literal subset used by the harness policy symbols."""

    def __init__(self, assignments: Mapping[str, ast.expr]) -> None:
        self.assignments = dict(assignments)
        self._active: set[str] = set()

    def value(self, node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return [self.value(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self.value(item) for item in node.elts)
        if isinstance(node, ast.Set):
            return {self.value(item) for item in node.elts}
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise HarnessSyncError("static policy dicts cannot use unpacking")
            return {
                self.value(key): self.value(value)
                for key, value in zip(node.keys, node.values)
                if key is not None  # narrowed after the explicit unpacking guard
            }
        if isinstance(node, ast.Name):
            if node.id not in self.assignments:
                raise HarnessSyncError(f"static symbol {node.id!r} is not assigned")
            if node.id in self._active:
                raise HarnessSyncError(f"cyclic static assignment involving {node.id!r}")
            self._active.add(node.id)
            try:
                return self.value(self.assignments[node.id])
            finally:
                self._active.remove(node.id)
        if self._is_environ_get(node):
            call = node
            assert isinstance(call, ast.Call)
            if len(call.args) < 2:
                raise HarnessSyncError(
                    "policy os.environ.get calls must provide a static default"
                )
            return self.value(call.args[1])
        raise HarnessSyncError(
            f"unsupported static policy expression: {ast.dump(node, include_attributes=False)}"
        )

    @staticmethod
    def _is_environ_get(node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "environ"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "os"
        )


def _simple_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    assignments: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            assignments[node.target.id] = node.value
    return assignments


def _required_expr(assignments: Mapping[str, ast.expr], name: str) -> ast.expr:
    try:
        return assignments[name]
    except KeyError as exc:
        raise HarnessSyncError(
            f"required harness.py symbol {name!r} is missing; the sync parser contract must be updated"
        ) from exc


def _subscript_key(target: ast.expr, collection: str) -> str | None:
    if not isinstance(target, ast.Subscript):
        return None
    if not isinstance(target.value, ast.Name) or target.value.id != collection:
        return None
    key = target.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    return None


def _all_assignments(tree: ast.Module) -> list[ast.Assign | ast.AnnAssign]:
    nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))]
    return sorted(nodes, key=lambda node: (node.lineno, node.col_offset))


def _assignment_targets(node: ast.Assign | ast.AnnAssign) -> Iterable[ast.expr]:
    if isinstance(node, ast.Assign):
        return node.targets
    return (node.target,)


def _assignment_value(node: ast.Assign | ast.AnnAssign) -> ast.expr | None:
    return node.value


def _dict_items(node: ast.expr, symbol: str) -> list[tuple[str, ast.expr]]:
    if not isinstance(node, ast.Dict):
        raise HarnessSyncError(f"{symbol} must remain a static dict literal")
    result: list[tuple[str, ast.expr]] = []
    for key_node, value_node in zip(node.keys, node.values):
        if key_node is None:
            raise HarnessSyncError(f"{symbol} cannot use dict unpacking")
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            raise HarnessSyncError(f"{symbol} keys must remain string literals")
        result.append((key_node.value, value_node))
    return result


def _backend_value_parts(node: ast.expr, name: str) -> tuple[ast.expr, str]:
    if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) != 3:
        raise HarnessSyncError(
            f"BACKENDS[{name!r}] must remain an (argv, edits_files, kind) tuple"
        )
    argv, edits_files, kind = node.elts
    if not isinstance(argv, (ast.List, ast.Tuple)):
        raise HarnessSyncError(f"BACKENDS[{name!r}] argv must remain a static list/tuple")
    if not isinstance(edits_files, ast.Constant) or not isinstance(edits_files.value, bool):
        raise HarnessSyncError(f"BACKENDS[{name!r}] edits_files must remain a boolean literal")
    if not isinstance(kind, ast.Constant) or not isinstance(kind.value, str):
        raise HarnessSyncError(f"BACKENDS[{name!r}] kind must remain a string literal")
    return argv, kind.value


def _model_from_argv(
    argv: ast.expr,
    *,
    evaluator: _StaticEvaluator,
) -> tuple[str | None, str]:
    assert isinstance(argv, (ast.List, ast.Tuple))
    items = argv.elts
    for index, item in enumerate(items[:-1]):
        if isinstance(item, ast.Constant) and item.value in {"-m", "--model"}:
            model_node = items[index + 1]
            source = ast.unparse(model_node)
            try:
                value = evaluator.value(model_node)
            except HarnessSyncError:
                # For example CATALOG["or-free-name"]["id"] is a static
                # external catalog reference, but harness.py itself does not
                # contain the concrete model ID. Preserve the expression and
                # do not import that catalog or execute the harness.
                if _expression_root_name(model_node) == "CATALOG":
                    return None, source
                raise
            if value in {None, ""}:
                return None, source
            if not isinstance(value, str):
                raise HarnessSyncError(f"backend model {source!r} did not resolve to a string")
            return value, source
    return None, "provider-default"


def _expression_root_name(node: ast.expr) -> str | None:
    current = node
    while isinstance(current, (ast.Subscript, ast.Attribute)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _backend_env_models(
    tree: ast.Module,
    evaluator: _StaticEvaluator,
) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for node in _all_assignments(tree):
        value = _assignment_value(node)
        if value is None:
            continue
        for target in _assignment_targets(node):
            backend = _subscript_key(target, "BACKEND_ENV")
            if backend is None:
                continue
            if not isinstance(value, ast.Dict):
                raise HarnessSyncError(f"BACKEND_ENV[{backend!r}] must remain a static dict literal")
            for key_node, value_node in zip(value.keys, value.values):
                if (
                    isinstance(key_node, ast.Constant)
                    and key_node.value == "ANTHROPIC_DEFAULT_SONNET_MODEL"
                ):
                    source = ast.unparse(value_node)
                    model = evaluator.value(value_node)
                    if not isinstance(model, str) or not model:
                        raise HarnessSyncError(
                            f"BACKEND_ENV[{backend!r}] Sonnet model must resolve to a non-empty string"
                        )
                    result[backend] = (model, source)
    return result


def _backend_nodes(tree: ast.Module, assignments: Mapping[str, ast.expr]) -> dict[str, ast.expr]:
    result = dict(_dict_items(_required_expr(assignments, "BACKENDS"), "BACKENDS"))
    for node in _all_assignments(tree):
        value = _assignment_value(node)
        if value is None:
            continue
        for target in _assignment_targets(node):
            backend = _subscript_key(target, "BACKENDS")
            if backend is None:
                continue
            if backend in result:
                raise HarnessSyncError(
                    f"BACKENDS[{backend!r}] is assigned more than once; refusing ambiguous policy"
                )
            result[backend] = value
    if not result:
        raise HarnessSyncError("BACKENDS is empty")
    return result


def _string_set(evaluator: _StaticEvaluator, node: ast.expr, symbol: str) -> frozenset[str]:
    value = evaluator.value(node)
    if not isinstance(value, (set, frozenset, tuple, list)) or not all(
        isinstance(item, str) for item in value
    ):
        raise HarnessSyncError(f"{symbol} must resolve to a collection of backend names")
    return frozenset(value)


def _classify_kind(name: str, kind: str, api_backends: frozenset[str]) -> tuple[str, str]:
    if name in api_backends:
        if kind != "API$$":
            raise HarnessSyncError(
                f"paid backend {name!r} has unexpected kind {kind!r}; update the sync classifier"
            )
        return "paid_api", "paid"
    if kind in {"sub", "sub-free", "sub-anthropic", "sub-glm", "sub-kimi"}:
        return "subscription", "free"
    if kind in {"or-free", "omni-free"}:
        return "free_service", "free"
    raise HarnessSyncError(
        f"backend {name!r} has unknown kind {kind!r}; update the sync classifier explicitly"
    )


def parse_harness_source(source: str, *, filename: str = "harness.py") -> HarnessPolicyFacts:
    """Extract the static policy contract without importing ``harness.py``."""

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise HarnessSyncError(f"cannot parse {filename}: {exc}") from exc
    assignments = _simple_assignments(tree)
    evaluator = _StaticEvaluator(assignments)

    api_backends = _string_set(
        evaluator, _required_expr(assignments, "API_BACKENDS"), "API_BACKENDS"
    )
    auto_disabled = _string_set(
        evaluator,
        _required_expr(assignments, "AUTO_DISABLED_BACKENDS"),
        "AUTO_DISABLED_BACKENDS",
    )
    env_models = _backend_env_models(tree, evaluator)
    backend_nodes = _backend_nodes(tree, assignments)

    unknown_paid = api_backends.difference(backend_nodes)
    if unknown_paid:
        raise HarnessSyncError(f"API_BACKENDS references unknown backends: {sorted(unknown_paid)}")
    unknown_disabled = auto_disabled.difference(backend_nodes)
    if unknown_disabled:
        raise HarnessSyncError(
            f"AUTO_DISABLED_BACKENDS references unknown backends: {sorted(unknown_disabled)}"
        )

    backends: dict[str, BackendFact] = {}
    for name, node in sorted(backend_nodes.items()):
        argv, kind = _backend_value_parts(node, name)
        model, model_source = _model_from_argv(argv, evaluator=evaluator)
        if model is None and model_source == "provider-default" and name in env_models:
            model, env_source = env_models[name]
            model_source = f"BACKEND_ENV[{name!r}].ANTHROPIC_DEFAULT_SONNET_MODEL ({env_source})"
        funding, billing_class = _classify_kind(name, kind, api_backends)
        backends[name] = BackendFact(
            name=name,
            kind=kind,
            funding=funding,
            billing_class=billing_class,
            model=model,
            model_source=model_source,
            automatic_enabled=name not in auto_disabled and name not in api_backends,
        )

    # The existing router structurally rejects GLM from automatic routes.  If
    # harness.py ever enables or removes it, regeneration must stop for a human
    # compatibility decision rather than claiming the two policies are synced.
    if "glm" not in backends:
        raise HarnessSyncError(
            "harness.py no longer declares the expected GLM backend; review the router GLM invariant"
        )
    if "glm" not in auto_disabled:
        raise HarnessSyncError(
            "harness.py enables GLM for automatic routing, but RoutePolicy forbids it; human review required"
        )

    lane_value = evaluator.value(_required_expr(assignments, "LANE_TABLE"))
    if not isinstance(lane_value, dict) or not lane_value:
        raise HarnessSyncError("LANE_TABLE must resolve to a non-empty dict")
    lane_table: dict[str, tuple[str, ...]] = {}
    for lane, order in lane_value.items():
        if not isinstance(lane, str) or not isinstance(order, (tuple, list)) or not order:
            raise HarnessSyncError("LANE_TABLE entries must map lane strings to backend sequences")
        if not all(isinstance(item, str) for item in order):
            raise HarnessSyncError(f"LANE_TABLE[{lane!r}] contains a non-string backend")
        unknown = set(order).difference(backends)
        if unknown:
            raise HarnessSyncError(
                f"LANE_TABLE[{lane!r}] references unknown backends: {sorted(unknown)}"
            )
        lane_table[lane] = tuple(order)

    default_lane = evaluator.value(_required_expr(assignments, "DEFAULT_LANE"))
    if not isinstance(default_lane, str) or default_lane not in lane_table:
        raise HarnessSyncError("DEFAULT_LANE must name a lane present in LANE_TABLE")

    return HarnessPolicyFacts(
        backends=backends,
        lane_table=lane_table,
        default_lane=default_lane,
        auto_disabled_backends=auto_disabled,
    )


def read_harness_policy(path: str | Path) -> HarnessPolicyFacts:
    source_path = Path(path).expanduser()
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HarnessSyncError(f"cannot read harness source {source_path}: {exc}") from exc
    return parse_harness_source(source, filename=str(source_path))


def _quoted_tuple(values: Iterable[str]) -> str:
    items = tuple(values)
    if not items:
        return "()"
    if len(items) == 1:
        return f"({items[0]!r},)"
    return "(" + ", ".join(repr(item) for item in items) + ")"


def render_policy_module(facts: HarnessPolicyFacts) -> str:
    """Render deterministic, importable policy data from extracted facts."""

    lines = [
        '"""Generated from harness.py static routing policy. DO NOT EDIT BY HAND.\n',
        "\n",
        "Regenerate with ``python -m jainsons_llm_router.sync_from_harness``.\n",
        "The digest covers extracted policy facts only, not harness runtime logic.\n",
        '"""\n',
        "\n",
        "from __future__ import annotations\n",
        "\n",
        "from collections.abc import Iterable, Mapping\n",
        "from dataclasses import dataclass\n",
        "from types import MappingProxyType\n",
        "\n",
        "from ..errors import ConfigurationError\n",
        "from ..models import BillingClass, Candidate\n",
        "\n",
        "\n",
        "@dataclass(frozen=True)\n",
        "class HarnessBackendPolicy:\n",
        "    name: str\n",
        "    kind: str\n",
        "    funding: str\n",
        "    billing_class: BillingClass\n",
        "    model: str | None\n",
        "    model_source: str\n",
        "    automatic_enabled: bool\n",
        "\n",
        "\n",
        f"HARNESS_POLICY_SHA256 = {facts.policy_sha256!r}\n",
        f"DEFAULT_LANE = {facts.default_lane!r}\n",
        f"AUTO_DISABLED_BACKENDS = frozenset({_quoted_tuple(sorted(facts.auto_disabled_backends))})\n",
        "GLM_AUTOMATIC_DISABLED = 'glm' in AUTO_DISABLED_BACKENDS\n",
        "\n",
        "BACKENDS = MappingProxyType({\n",
    ]
    for name, fact in sorted(facts.backends.items()):
        billing = "BillingClass.FREE" if fact.billing_class == "free" else "BillingClass.PAID"
        lines.extend(
            [
                f"    {name!r}: HarnessBackendPolicy(\n",
                f"        name={name!r}, kind={fact.kind!r}, funding={fact.funding!r},\n",
                f"        billing_class={billing}, model={fact.model!r},\n",
                f"        model_source={fact.model_source!r}, automatic_enabled={fact.automatic_enabled!r},\n",
                "    ),\n",
            ]
        )
    lines.extend(["})\n", "\n", "LANE_BACKEND_ORDER = MappingProxyType({\n"])
    for lane, order in sorted(facts.lane_table.items()):
        lines.append(f"    {lane!r}: {_quoted_tuple(order)},\n")
    lines.extend(["})\n", "\n", "FREE_CANDIDATE_BACKENDS_BY_LANE = MappingProxyType({\n"])
    for lane, order in sorted(facts.free_candidate_backends_by_lane.items()):
        lines.append(f"    {lane!r}: {_quoted_tuple(order)},\n")
    lines.extend(
        [
            "})\n",
            "\n",
            "\n",
            "def free_candidate_backend_order(lane: str) -> tuple[str, ...]:\n",
            "    try:\n",
            "        return FREE_CANDIDATE_BACKENDS_BY_LANE[lane]\n",
            "    except KeyError as exc:\n",
            "        raise ConfigurationError(f'unknown harness lane: {lane}') from exc\n",
            "\n",
            "\n",
            "def order_free_candidates(\n",
            "    lane: str,\n",
            "    candidates_by_backend: Mapping[str, Candidate],\n",
            "    *,\n",
            "    allow_missing: Iterable[str] = (),\n",
            ") -> tuple[Candidate, ...]:\n",
            "    \"\"\"Apply harness order to deployment-owned free Candidate objects.\n",
            "\n",
            "    Missing harness backends fail unless explicitly acknowledged in\n",
            "    ``allow_missing``. Paid candidates, caps, approvals, adapter/account\n",
            "    wiring, and exact-model pins remain local to the consuming route.\n",
            "    \"\"\"\n",
            "\n",
            "    order = free_candidate_backend_order(lane)\n",
            "    allowed = frozenset(allow_missing)\n",
            "    unknown_allowed = allowed.difference(order)\n",
            "    if unknown_allowed:\n",
            "        raise ConfigurationError(\n",
            "            f'allow_missing contains backends outside lane {lane}: {sorted(unknown_allowed)}'\n",
            "        )\n",
            "    missing = [name for name in order if name not in candidates_by_backend and name not in allowed]\n",
            "    if missing:\n",
            "        raise ConfigurationError(\n",
            "            f'missing harness-derived free candidates for lane {lane}: {missing}'\n",
            "        )\n",
            "    selected: list[Candidate] = []\n",
            "    for name in order:\n",
            "        candidate = candidates_by_backend.get(name)\n",
            "        if candidate is None:\n",
            "            continue\n",
            "        if candidate.billing_class is not BillingClass.FREE:\n",
            "            raise ConfigurationError(f'harness backend {name} must map to a free Candidate')\n",
            "        if not candidate.zero_marginal_cost:\n",
            "            raise ConfigurationError(\n",
            "                f'harness backend {name} must prove zero marginal cost'\n",
            "            )\n",
            "        selected.append(candidate)\n",
            "    return tuple(selected)\n",
            "\n",
        ]
    )
    return "".join(lines)


def _drift_diff(current: str, expected: str, output_path: Path) -> str:
    diff = difflib.unified_diff(
        current.splitlines(),
        expected.splitlines(),
        fromfile=str(output_path),
        tofile=f"{output_path} (regenerated)",
        lineterm="",
    )
    return "\n".join(list(diff)[:80])


def sync(
    *,
    harness_path: str | Path = DEFAULT_HARNESS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    check: bool = False,
) -> int:
    facts = read_harness_policy(harness_path)
    rendered = render_policy_module(facts)
    output = Path(output_path).expanduser()
    if check:
        try:
            current = output.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"harness-sync: drift: generated policy is missing: {output}", file=sys.stderr)
            return 1
        except OSError as exc:
            raise HarnessSyncError(f"cannot read generated policy {output}: {exc}") from exc
        if current != rendered:
            print(
                "harness-sync: drift detected; run "
                "`python -m jainsons_llm_router.sync_from_harness`",
                file=sys.stderr,
            )
            print(_drift_diff(current, rendered, output), file=sys.stderr)
            return 1
        print(f"harness-sync: current ({output})")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise HarnessSyncError(f"cannot write generated policy {output}: {exc}") from exc
    print(f"harness-sync: wrote {output}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harness",
        type=Path,
        default=DEFAULT_HARNESS_PATH,
        help=f"harness.py source path (default: {DEFAULT_HARNESS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"generated policy path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 when the committed generated policy is stale",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return sync(harness_path=args.harness, output_path=args.output, check=args.check)
    except HarnessSyncError as exc:
        print(f"harness-sync: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
