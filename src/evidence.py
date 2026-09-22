"""The evidence envelope every tool - policy, state, or derived - returns
through.

The Phase 3 verifier checks a claim against an evidence_id without caring
which tool produced it, so every source is wrapped identically:

    {"evidence_id": "...", "content": ..., "source_type": "policy"|"state"|"derived"}

evidence_id follows the scheme in corpus/README.md:
  - policy chunk:  doc:<slug>#c<n>
  - state field:   state:<merchant_id>.<dotted.path>
  - derived fact:  derived:<merchant_id>.<dotted.path>

"derived" exists because the blind adversarial check showed the verifier
cannot check date arithmetic: it has no notion of "now" and its own prompt
forbids the outside knowledge (a calendar) that business-day counting
needs. Rather than loosen the verifier, the arithmetic moved into Python
(src/derived_facts.py), where it is deterministic and testable, and its
results became citable evidence like anything else.

This module is also the single enforcement point for the `_derivation`
convention (see corpus/README.md and Phase 0's fixture fixes): merchant
fixtures carry `_`-prefixed keys that are eval-only and must never reach
the agent. Rather than trust every tool author to remember to strip them,
`make_evidence` strips any `_`-prefixed key recursively before content
ever leaves this module - so a tool that forgets is still safe, and a
tool that bypasses this module entirely is the only way to leak one
(caught instead by tests exercising the tools directly).
"""

from __future__ import annotations

import re
from typing import Any, Literal

SourceType = Literal["policy", "state", "derived"]

_VALID_SOURCE_TYPES = ("policy", "state", "derived")

# doc:<slug>#c<n> - slug is a policy filename stem, n is a 1-indexed chunk
# number assigned in document order (see corpus/README.md).
_POLICY_ID_RE = re.compile(r"^doc:[a-z0-9-]+#c[1-9]\d*$")

# state:<merchant_id>[.<dotted.path>], where each path segment is an
# identifier optionally followed by one or more [index] accessors, e.g.
# state:merchant_3.transactions[12].status
#
# The dotted path is optional: a bare `state:merchant_3` cites the whole
# merchant record as one unit, which is what a bulk lookup like
# get_merchant_state returns. A tool that returns one specific field (e.g.
# get_settlement_schedule) should cite the narrower path instead - prefer
# the most specific id the tool actually has, since that's what the
# Phase 3 verifier checks a claim against.
_STATE_ID_RE = re.compile(
    r"^state:merchant_\d+(\.[a-zA-Z_][a-zA-Z0-9_]*(\[\d+\])?)*$"
)

# derived:<merchant_id>.<dotted.path> - same path grammar as state, but a
# required path (there is no "whole derived record" to cite; every derived
# fact is one named computation, e.g.
# derived:merchant_1.expected_settlement_date).
_DERIVED_ID_RE = re.compile(
    r"^derived:merchant_\d+(\.[a-zA-Z_][a-zA-Z0-9_]*(\[\d+\])?)+$"
)

_ID_PATTERNS = {
    "policy": _POLICY_ID_RE,
    "state": _STATE_ID_RE,
    "derived": _DERIVED_ID_RE,
}


class InvalidEvidenceId(ValueError):
    pass


def _strip_hidden(obj: Any) -> Any:
    """Recursively drop any dict key starting with `_`."""
    if isinstance(obj, dict):
        return {k: _strip_hidden(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_hidden(v) for v in obj]
    return obj


def _validate_evidence_id(evidence_id: str, source_type: SourceType) -> None:
    pattern = _ID_PATTERNS[source_type]
    if not pattern.match(evidence_id):
        raise InvalidEvidenceId(
            f"evidence_id {evidence_id!r} does not match the {source_type} "
            f"scheme documented in corpus/README.md"
        )


def make_evidence(evidence_id: str, content: Any, source_type: SourceType) -> dict:
    """Wrap a fact in the standard evidence envelope.

    Validates evidence_id against the scheme for its source_type and
    strips any `_`-prefixed key out of content, recursively, before
    returning.
    """
    if source_type not in _VALID_SOURCE_TYPES:
        raise ValueError(
            f"source_type must be one of {_VALID_SOURCE_TYPES}, got {source_type!r}"
        )
    _validate_evidence_id(evidence_id, source_type)
    return {
        "evidence_id": evidence_id,
        "content": _strip_hidden(content),
        "source_type": source_type,
    }
