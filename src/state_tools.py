"""Plain lookups over corpus/merchants/*.json. No LLM, no network calls.

Every successful lookup returns evidence through src.evidence.make_evidence
(source_type="state"), so a raw `_derivation` field can never reach a
caller - make_evidence strips it. A nonexistent merchant_id never returns
an empty success; it returns the explicit not-found shape from
`_not_found()`, structurally distinct from a real envelope (no
evidence_id/content/source_type keys at all), so a caller can't
accidentally treat "merchant doesn't exist" as "merchant exists with no
data".

Every elapsed-time computation belongs to the caller (agent/verifier),
using src.config.CORPUS_NOW - nothing here reads the real wall-clock date,
and tests/test_no_realtime_clock.py guards the whole src/ tree against
that creeping back in.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from src.evidence import make_evidence

MERCHANTS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "merchants"


def _not_found(merchant_id: str) -> dict:
    return {
        "not_found": True,
        "merchant_id": merchant_id,
        "message": f"No merchant fixture exists for merchant_id={merchant_id!r}",
    }


def is_not_found(result: Any) -> bool:
    """True if a tool result is the not-found sentinel rather than a real
    evidence envelope or list of envelopes."""
    return isinstance(result, dict) and result.get("not_found") is True


def _load_merchant(merchant_id: str) -> dict | None:
    path = MERCHANTS_DIR / f"{merchant_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_merchant_state(merchant_id: str) -> dict:
    """The merchant's entire fixture record as one evidence envelope.

    Evidence id is the bare `state:<merchant_id>` form (see corpus/
    README.md): this is a bulk lookup, not a citation of one field.
    """
    merchant = _load_merchant(merchant_id)
    if merchant is None:
        return _not_found(merchant_id)
    return make_evidence(f"state:{merchant_id}", merchant, "state")


def get_settlement_schedule(merchant_id: str) -> dict:
    """The merchant's settlement_schedule sub-object as one evidence envelope."""
    merchant = _load_merchant(merchant_id)
    if merchant is None:
        return _not_found(merchant_id)
    return make_evidence(
        f"state:{merchant_id}.settlement_schedule",
        merchant["settlement_schedule"],
        "state",
    )


def get_disputes(merchant_id: str) -> list[dict] | dict:
    """One evidence envelope per dispute on the merchant's account.

    Returns an empty list (not the not-found sentinel) for a real merchant
    with no disputes - that is a legitimate fact, not an error. Only a
    merchant_id that doesn't resolve to a fixture at all returns
    the not-found sentinel.
    """
    merchant = _load_merchant(merchant_id)
    if merchant is None:
        return _not_found(merchant_id)
    return [
        make_evidence(f"state:{merchant_id}.disputes[{i}]", dispute, "state")
        for i, dispute in enumerate(merchant.get("disputes", []))
    ]


def get_transactions(merchant_id: str, filters: dict | None = None) -> list[dict] | dict:
    """One evidence envelope per transaction matching `filters`.

    `filters` (all optional, combined with AND):
      - status: exact match, e.g. "captured" | "failed" | "refunded"
      - method: exact match, e.g. "card" | "upi" | "netbanking" | "wallet"
      - failure_code: exact match, e.g. "RISK_BLOCK"
      - start_date / end_date: inclusive "YYYY-MM-DD" bounds on the
        transaction's own date field

    evidence_id indexes into the merchant's *original, unfiltered*
    transaction list, so a citation stays meaningful regardless of which
    filters produced it.
    """
    merchant = _load_merchant(merchant_id)
    if merchant is None:
        return _not_found(merchant_id)

    filters = filters or {}
    start_date = filters.get("start_date")
    end_date = filters.get("end_date")

    results = []
    for i, txn in enumerate(merchant.get("transactions", [])):
        if "status" in filters and txn.get("status") != filters["status"]:
            continue
        if "method" in filters and txn.get("method") != filters["method"]:
            continue
        if "failure_code" in filters and txn.get("failure_code") != filters["failure_code"]:
            continue
        if start_date and txn["date"] < start_date:
            continue
        if end_date and txn["date"] > end_date:
            continue
        results.append(
            make_evidence(f"state:{merchant_id}.transactions[{i}]", txn, "state")
        )
    return results


def _name_similarity(query: str, business_name: str) -> float:
    """difflib's ratio penalizes length mismatches heavily, so a short
    partial name ("Zenith") scores low against a long full name ("Zenith
    Fitness Equipment") even though it's an exact, confident substring
    match. Boost pure substring containment separately rather than
    trusting ratio alone for this tool's actual use case: matching a
    ticket's casual, partial mention of a business against its full
    registered name.
    """
    q = query.strip().lower()
    n = business_name.strip().lower()
    if not q:
        return 0.0
    ratio = difflib.SequenceMatcher(None, q, n).ratio()
    if q in n or n in q:
        shorter, longer = (q, n) if len(q) <= len(n) else (n, q)
        containment = 0.5 + 0.5 * (len(shorter) / len(longer))
        ratio = max(ratio, containment)
    return ratio


def find_merchant_by_name(query: str, top_n: int = 5) -> list[dict]:
    """Fuzzy-match `query` against every merchant's business_name.

    Returns up to top_n evidence envelopes, most similar first, each
    wrapping {merchant_id, business_name, score}. This is a name matcher,
    not a semantic one: it will confidently resolve "Zenith" or "this is
    Zenith" against "Zenith Fitness Equipment", but it will NOT resolve a
    business *description* with no name in it ("we run a tiffin delivery
    service") to "Anjali Tiffin Services" with any real confidence - that
    needs reasoning beyond a plain lookup, which is out of scope here.

    Never returns the not-found sentinel: an empty or low-scoring result
    list is itself the answer (no good match), not a lookup failure.
    """
    candidates = []
    for path in sorted(MERCHANTS_DIR.glob("merchant_*.json")):
        merchant = json.loads(path.read_text(encoding="utf-8"))
        score = _name_similarity(query, merchant["business_name"])
        candidates.append((score, merchant["merchant_id"], merchant["business_name"]))

    candidates.sort(key=lambda c: -c[0])

    return [
        make_evidence(
            f"state:{merchant_id}",
            {"merchant_id": merchant_id, "business_name": business_name, "score": round(score, 4)},
            "state",
        )
        for score, merchant_id, business_name in candidates[:top_n]
    ]
