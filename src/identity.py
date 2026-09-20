"""Merchant identification: resolve who is actually asking, before any
diagnosis happens.

Priority order (most to least reliable), per the design rationale that a
deterministic lookup beats a fuzzy guess, which in turn beats trusting an
unverified claim:

  1. resolve_reference() on any dispute/transaction/refund/mandate/
     customer-ref id found in the ticket body. Exact match against every
     merchant's fixture - not a guess.
  2. find_merchant_by_name() on business-name-shaped text in the body.
     A fuzzy match, scored, only trusted above NAME_MATCH_THRESHOLD.
  3. the ticket's own submitted merchant_id field.

The submitted merchant_id is a CLAIM, not a fact. When step 1 or 2
resolves to a merchant *different* from the claim, that is a MISMATCH:
answering from the claimed merchant's state would expose one merchant's
account data to another. This is a named risk class the caller must
refuse to proceed past, not a "best guess, slightly wrong" outcome to
paper over.
"""

from __future__ import annotations

import re
from typing import Literal

from src.state_tools import find_merchant_by_name, get_merchant_state, is_not_found, resolve_reference

Outcome = Literal["CONFIRMED", "UNCORROBORATED", "MISMATCH", "UNIDENTIFIABLE"]

# Empirically checked (see tests/test_identity.py): a genuine partial-name
# mention like "Zenith" against "Zenith Fitness Equipment" scores 0.625; an
# extracted full name missing punctuation ("Trailblazer Holiday Co" vs
# "Trailblazer Holiday Co.") scores 0.978. Below this, a match is treated
# as noise rather than corroboration.
NAME_MATCH_THRESHOLD = 0.6

_REFERENCE_RE = re.compile(r"\b(?:disp|txn|refund|mandate|cust)_\d+(?:_\d+)?\b")

# Case-insensitive cue phrase, case-SENSITIVE captured name (scoped inline
# flag) - "this is X" / "I run X" style self-identification. Deliberately
# narrow: matching on ordinary sentence capitalization elsewhere produced
# large false-positive captures during development (see conversation/
# commit history) - a plain "capitalized word" heuroistic swallows things
# like "Please investigate immediately" whenever a ticket happens to have
# two capitalized words in a row.
_NAME_CUE_RE = re.compile(
    r"(?i:this is|i am|i'm|we are|we're|i run|we run|"
    r"my (?:business|company|shop|store) is)"
    r"\s+([A-Z][a-z']*(?:\s+[A-Z][a-z']*){0,4})"
)

# Fallback: a bare multi-word Title Case run with no cue phrase at all
# (e.g. a business name mentioned mid-sentence). Requires an actual
# lowercase letter after each word's initial capital, which excludes
# ALL-CAPS emphasis ("FIX THIS", "REJECTED AGAIN") and short interjections.
_TITLECASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b")

_STOPWORDS = {
    "the", "this", "that", "my", "your", "please", "but", "as", "can",
    "and", "for", "if", "hi", "hello", "regards", "two", "one", "also",
    "ok", "okay", "so",
}


def _extract_references(body: str) -> list[str]:
    seen = []
    for match in _REFERENCE_RE.findall(body):
        if match not in seen:
            seen.append(match)
    return seen


def _extract_name_candidates(body: str) -> list[str]:
    candidates = []
    for pattern in (_NAME_CUE_RE, _TITLECASE_RE):
        for match in pattern.findall(body):
            words = match.split()
            if all(w.lower() in _STOPWORDS for w in words):
                continue
            if match not in candidates:
                candidates.append(match)
    return candidates


def identify_merchant(ticket: dict) -> dict:
    """Identify who a ticket is actually from.

    `ticket` needs at least {"merchant_id": str | None, "body": str}.

    Returns:
      {
        "outcome": one of CONFIRMED / UNCORROBORATED / MISMATCH / UNIDENTIFIABLE,
        "identified_merchant_id": the merchant to actually use, or None,
        "claimed_merchant_id": ticket["merchant_id"], unchanged,
        "evidence": [evidence envelopes backing the identification],
        "signals": {"references": [...], "names": [...], "reference_matches": [...], "name_matches": [...]},
      }
    """
    claimed_id = ticket.get("merchant_id")
    body = ticket.get("body", "")

    references = _extract_references(body)
    names = _extract_name_candidates(body)

    evidence = []
    reference_matches: list[str] = []  # merchant_ids, in order found
    for ref in references:
        result = resolve_reference(ref)
        if not is_not_found(result):
            evidence.append(result)
            mid = result["content"]["merchant_id"]
            if mid not in reference_matches:
                reference_matches.append(mid)

    name_matches: list[str] = []
    if not reference_matches:  # step 2 only matters if step 1 found nothing
        for name in names:
            candidates = find_merchant_by_name(name, top_n=1)
            if candidates and candidates[0]["content"]["score"] >= NAME_MATCH_THRESHOLD:
                evidence.append(candidates[0])
                mid = candidates[0]["content"]["merchant_id"]
                if mid not in name_matches:
                    name_matches.append(mid)

    corroborating_ids = reference_matches or name_matches

    signals = {
        "references": references,
        "names": names,
        "reference_matches": reference_matches,
        "name_matches": name_matches,
    }

    claimed_id_exists = claimed_id is not None and not is_not_found(get_merchant_state(claimed_id))

    if len(set(corroborating_ids)) > 1:
        # Two independent signals in the same ticket corroborate two
        # DIFFERENT merchants - an internal contradiction, not just a
        # mismatch against the claim. Rare (no ticket in this corpus hits
        # it), but silently picking one would be exactly the kind of
        # unearned confidence this module exists to prevent.
        return {
            "outcome": "UNIDENTIFIABLE",
            "identified_merchant_id": None,
            "claimed_merchant_id": claimed_id,
            "evidence": evidence,
            "signals": signals,
        }

    if not corroborating_ids:
        if claimed_id is None or not claimed_id_exists:
            outcome: Outcome = "UNIDENTIFIABLE"
            identified = None
        else:
            outcome = "UNCORROBORATED"
            identified = claimed_id
        return {
            "outcome": outcome,
            "identified_merchant_id": identified,
            "claimed_merchant_id": claimed_id,
            "evidence": evidence,
            "signals": signals,
        }

    corroborated_id = corroborating_ids[0]
    if claimed_id is None or claimed_id == corroborated_id:
        outcome = "CONFIRMED"
        identified = corroborated_id
    else:
        outcome = "MISMATCH"
        identified = corroborated_id

    return {
        "outcome": outcome,
        "identified_merchant_id": identified,
        "claimed_merchant_id": claimed_id,
        "evidence": evidence,
        "signals": signals,
    }
