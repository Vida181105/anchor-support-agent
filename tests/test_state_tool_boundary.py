"""Enforces the `_derivation` boundary at the one place it actually matters:
the return value of a state tool the agent calls.

corpus/README.md and the merchant fixtures establish the convention that any
key beginning with `_` (e.g. `_derivation`) is eval-only and must never reach
the agent. Nothing enforces that yet because no state tool exists — this
test is written against the interface Phase 1 is expected to add
(`src/state_tools.py`, exposing `get_merchant_state(merchant_id: str) -> dict`)
so that it starts failing the moment that tool exists and leaks a hidden key,
instead of silently passing forever because the import fails.
"""

import glob
import json

import pytest

STATE_TOOLS_MODULE = "src.state_tools"
STATE_TOOLS_FUNCTION = "get_merchant_state"


def _find_hidden_keys(obj, path=""):
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k.startswith("_"):
                hits.append(p)
            hits.extend(_find_hidden_keys(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(_find_hidden_keys(v, f"{path}[{i}]"))
    return hits


def _load_state_tool():
    try:
        module = __import__(STATE_TOOLS_MODULE, fromlist=[STATE_TOOLS_FUNCTION])
    except ImportError:
        return None
    return getattr(module, STATE_TOOLS_FUNCTION, None)


def test_state_tool_never_returns_hidden_keys():
    get_merchant_state = _load_state_tool()
    if get_merchant_state is None:
        pytest.skip(
            f"no state tools yet — implement {STATE_TOOLS_MODULE}.{STATE_TOOLS_FUNCTION}"
            f"(merchant_id) and this test will start enforcing the _derivation boundary "
            f"instead of skipping"
        )

    merchant_ids = [
        json.load(open(f))["merchant_id"]
        for f in sorted(glob.glob("corpus/merchants/merchant_*.json"))
    ]
    assert merchant_ids, "no merchant fixtures found to test against"

    violations = {}
    for merchant_id in merchant_ids:
        result = get_merchant_state(merchant_id)
        hits = _find_hidden_keys(result)
        if hits:
            violations[merchant_id] = hits

    assert not violations, f"state tool leaked hidden (_-prefixed) keys: {violations}"
