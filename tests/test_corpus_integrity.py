"""Structural checks on the corpus that don't require the retrieval/agent
layer to exist yet: no answer leakage in merchant fixtures, and no
hardcoded current-date reasoning that CORPUS_NOW would silently drift out
of sync with.

Once the state tools exist (Phase 1+), add a companion test asserting the
tool layer itself strips every `_`-prefixed key before returning state to
the agent — this file only checks the corpus files on disk.
"""

import glob
import json
import re
from datetime import datetime
from pathlib import Path

from src.config import CORPUS_NOW

ROOT = Path(__file__).resolve().parent.parent
CORPUS_MERCHANTS = ROOT / "corpus" / "merchants"
CORPUS_TICKETS = ROOT / "corpus" / "tickets"

# Matches CORPUS_NOW's date, or generic "as of <date>" elapsed-time language,
# which is exactly the pattern that goes stale if CORPUS_NOW ever moves.
ASOF_PATTERN = re.compile(r"\bas of\b", re.IGNORECASE)


def _walk_strings(obj, path, in_hidden=False):
    """Yield (path, value, in_hidden) for every string leaf in a JSON tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            hidden = in_hidden or k.startswith("_")
            yield from _walk_strings(v, f"{path}.{k}", hidden)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{path}[{i}]", in_hidden)
    elif isinstance(obj, str):
        yield path, obj, in_hidden


def _merchant_files():
    return sorted(glob.glob(str(CORPUS_MERCHANTS / "merchant_*.json")))


def test_no_note_fields_outside_derivation():
    """`note`/`notes` were the leak vector: prose stating the diagnosis
    directly. Any narrative explanation must live under a `_`-prefixed key
    (eval-only, never returned by a state tool) instead.
    """
    violations = []
    for f in _merchant_files():
        m = json.load(open(f))
        for path, _value, in_hidden in _walk_strings(m, m["merchant_id"]):
            leaf_key = path.rsplit(".", 1)[-1].split("[")[0]
            if leaf_key in ("note", "notes") and not in_hidden:
                violations.append(path)
    assert not violations, f"note/notes field(s) outside _derivation: {violations}"


def test_no_hardcoded_asof_narrative_outside_derivation():
    """Elapsed-time reasoning ("as of <date>, N days have passed...") must
    be computed from CORPUS_NOW at query time, not baked into a fixture as
    prose - otherwise it silently rots the moment CORPUS_NOW moves.
    """
    violations = []
    for f in _merchant_files():
        m = json.load(open(f))
        for path, value, in_hidden in _walk_strings(m, m["merchant_id"]):
            if in_hidden:
                continue
            if ASOF_PATTERN.search(value):
                violations.append((path, value))
    assert not violations, f"hardcoded 'as of' narrative outside _derivation: {violations}"


def test_no_ticket_timestamps_after_corpus_now():
    """A ticket dated after CORPUS_NOW is a future event by construction -
    guards against corpus edits (like ticket_018's original 2026-09-20
    timestamp) drifting past the single canonical "now".
    """
    violations = []
    for f in sorted(glob.glob(str(CORPUS_TICKETS / "ticket_*.json"))):
        t = json.load(open(f))
        ts = datetime.fromisoformat(t["timestamp"])
        if ts > CORPUS_NOW:
            violations.append((f, t["timestamp"]))
    assert not violations, f"ticket(s) timestamped after CORPUS_NOW: {violations}"


def test_derivation_fields_are_hidden_by_convention():
    """Every `_derivation` key must actually be underscore-prefixed at the
    key it's stored under (sanity check on the convention itself).
    """
    for f in _merchant_files():
        m = json.load(open(f))

        def check(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "_derivation":
                        assert isinstance(v, str) and v, f"{f}: empty _derivation"
                    check(v)
            elif isinstance(obj, list):
                for v in obj:
                    check(v)

        check(m)
