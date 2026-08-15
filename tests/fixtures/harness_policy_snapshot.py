"""Small harness.py-shaped fixture containing static policy declarations only."""

import os

BEST_MODEL = os.environ.get("HARNESS_BEST_MODEL", "")
CODEX_MODEL = os.environ.get("HARNESS_CODEX_MODEL", "gpt-fixture-main")
GLM_MODEL = os.environ.get("HARNESS_GLM_MODEL", "glm-fixture")
KIMI_MODEL = os.environ.get("HARNESS_KIMI_MODEL", "kimi-fixture")

CATALOG = {"or-free-fixture": {"id": "external/catalog-model"}}

BACKENDS = {
    "agy": (["agy", "-p", "{task}"], True, "sub-free"),
    "codex": (["codex", "-m", CODEX_MODEL, "exec", "{task}"], True, "sub"),
    "codex-sol": (["codex", "-m", "gpt-fixture-hard", "exec", "{task}"], True, "sub"),
    "codex-luna": (["codex", "-m", "gpt-fixture-fast", "exec", "{task}"], True, "sub"),
    "or-best": (["llm", "-m", BEST_MODEL, "{task}"], False, "API$$"),
    "or-free-fixture": (
        ["llm", "-m", CATALOG["or-free-fixture"]["id"], "{task}"],
        False,
        "or-free",
    ),
    "omni-fast": (["llm", "-m", "omni-fixture", "{task}"], False, "omni-free"),
}

if os.environ.get("HARNESS_DISABLE_CLAUDE_BACKEND") != "1":
    BACKENDS["claude"] = (["claude", "-p", "{task}"], True, "sub-anthropic")
    BACKENDS["claude-opus"] = (
        ["claude", "-p", "--model", "claude-fixture-opus", "{task}"],
        True,
        "sub-anthropic",
    )

BACKEND_ENV: dict[str, dict] = {}
if "fixture-zai-key":
    BACKENDS["glm"] = (["claude", "-p", "{task}"], True, "sub-glm")
    BACKEND_ENV["glm"] = {
        "ANTHROPIC_DEFAULT_SONNET_MODEL": GLM_MODEL,
    }

if "fixture-kimi-key":
    BACKENDS["kimi"] = (["claude", "-p", "{task}"], True, "sub-kimi")
    BACKEND_ENV["kimi"] = {
        "ANTHROPIC_DEFAULT_SONNET_MODEL": KIMI_MODEL,
    }

API_BACKENDS = {"or-best"}
LANE_TABLE = {
    "research": ("agy", "codex"),
    "code": ("codex", "agy"),
    "image_gen": ("agy", "agy"),
    "planning": ("claude", "codex"),
}
DEFAULT_LANE = "research"
AUTO_DISABLED_BACKENDS = {"glm", "kimi"}
