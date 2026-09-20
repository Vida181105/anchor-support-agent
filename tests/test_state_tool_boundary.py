"""Enforces the `_derivation` boundary at the mechanism every state tool is
required to route through: src.evidence.make_evidence.

corpus/README.md and the merchant fixtures establish the convention that any
key beginning with `_` (e.g. `_derivation`) is eval-only and must never reach
the agent. Rather than wait for src/state_tools.py to exist to prove this
(and skip vacuously until then), this test feeds every real merchant
fixture - which already has `_derivation` at several nesting depths, per
Phase 0's fixes - through make_evidence directly and asserts none of it
survives. That is the actual guarantee: any tool built on top of
make_evidence inherits it for free.

This does not, by itself, prove a given tool *uses* make_evidence instead
of returning a raw dict. tests/test_state_tools.py (added alongside the
real tools) closes that gap by calling the tools themselves.
"""

import glob
import json

from src.evidence import make_evidence


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


def _merchant_fixtures():
    files = sorted(glob.glob("corpus/merchants/merchant_*.json"))
    assert files, "no merchant fixtures found to test against"
    return [json.load(open(f)) for f in files]


def test_real_fixtures_contain_hidden_keys_before_wrapping():
    """Sanity check on the test itself: if this fails, the fixtures no
    longer exercise the boundary at all and the assertion below would pass
    vacuously.
    """
    total_hits = sum(len(_find_hidden_keys(m)) for m in _merchant_fixtures())
    assert total_hits > 0, (
        "no _-prefixed keys found in any merchant fixture - this test would "
        "pass even with no stripping logic at all"
    )


def test_make_evidence_strips_every_real_fixture_clean():
    violations = {}
    for merchant in _merchant_fixtures():
        envelope = make_evidence(
            f"state:{merchant['merchant_id']}", merchant, "state"
        )
        hits = _find_hidden_keys(envelope["content"])
        if hits:
            violations[merchant["merchant_id"]] = hits
    assert not violations, f"hidden keys survived make_evidence: {violations}"
