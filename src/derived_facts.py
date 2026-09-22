"""Deterministic derived facts: date arithmetic done in Python, not by a
model, and returned as citable evidence.

Why this exists. The blind adversarial check (eval/adversarial_blind.json,
case adv_1) put a business-day off-by-one in front of the verifier and the
verifier did not catch it - correctly, and for two structural reasons that
no prompt change could fix:

  1. The verifier has no notion of "now". src/verifier.py's prompt
     interpolates only the claim text and the cited evidence content, so a
     claim like "the payout is running a day behind schedule" is not
     checkable at all.
  2. Counting business days from a bare ISO date requires knowing what
     weekday 2026-09-15 falls on. That is a calendar fact, not something
     present in the evidence, and the verifier's own prompt forbids the
     outside knowledge that would supply it.

An LLM should not be counting business days anyway. So the arithmetic
happens here - deterministic, unit-tested, weekend-aware - and its results
become evidence with ids like `derived:merchant_1.expected_settlement_date`,
carried in the same envelope as policy and state.

Every derived fact's content is self-describing: it carries the inputs it
was computed from, the method in words, and `as_of` (CORPUS_NOW) where the
computation is relative to now. That matters because the verifier sees the
content and nothing else - it has to be able to confirm a claim against
this fact without needing any context the envelope doesn't carry.

Business days here are Monday-Friday. The policy KB also excludes bank
holidays (corpus/policy/settlement-cycles.md#c1), but the corpus defines no
holiday calendar, so none is applied - stated explicitly in each fact's
`method` rather than silently assumed.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from src.config import CORPUS_NOW
from src.evidence import make_evidence
from src.state_tools import _load_merchant, _not_found, is_not_found  # noqa: F401

# From corpus/policy/chargebacks-disputes.md#c4: the issuing bank typically
# takes 30-45 days to decide once evidence is submitted. Held here as the
# parameters of the window check, and echoed into the fact's own content so
# a claim citing this fact can be checked without the policy chunk too.
DISPUTE_REVIEW_WINDOW_DAYS = (30, 45)

_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def is_business_day(d: date) -> bool:
    """Monday-Friday. No holiday calendar exists in this corpus."""
    return d.weekday() < 5


def add_business_days(start: date, n: int) -> date:
    """Advance `n` business days from `start`, skipping weekends.

    Day counting is exclusive of `start`: add_business_days(Tue, 3) lands
    on Friday (Wed=1, Thu=2, Fri=3), which is the T+3 reading used by
    corpus/policy/settlement-cycles.md#c1. A `start` that is itself a
    weekend still counts forward from the next business day.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    current = start
    remaining = n
    while remaining > 0:
        current += timedelta(days=1)
        if is_business_day(current):
            remaining -= 1
    return current


def business_days_between(start: date, end: date) -> int:
    """Business days strictly after `start` up to and including `end`.

    The inverse of add_business_days: business_days_between(d,
    add_business_days(d, n)) == n. Returns 0 if `end` <= `start`.
    """
    if end <= start:
        return 0
    count = 0
    current = start
    while current < end:
        current += timedelta(days=1)
        if is_business_day(current):
            count += 1
    return count


def _corpus_today() -> date:
    return CORPUS_NOW.date()


def _parse(d: str) -> date:
    return date.fromisoformat(d)


def _cycle_business_days(cycle: str) -> int:
    """'T+3' -> 3. Raises on anything that isn't a T+<n> cycle."""
    if not cycle.startswith("T+"):
        raise ValueError(f"unrecognised settlement cycle: {cycle!r}")
    return int(cycle[2:])


def _weekday_name(d: date) -> str:
    return _WEEKDAY_NAMES[d.weekday()]


def _facts_for_settlement(merchant_id: str, merchant: dict) -> list[tuple[str, Any]]:
    schedule = merchant.get("settlement_schedule", {})
    captured_raw = schedule.get("oldest_unsettled_batch_captured_date")
    cycle = schedule.get("cycle")
    if not captured_raw or not cycle:
        return []

    captured = _parse(captured_raw)
    n = _cycle_business_days(cycle)
    expected = add_business_days(captured, n)
    today = _corpus_today()
    elapsed = business_days_between(captured, today)

    steps = []
    cursor = captured
    for i in range(n):
        cursor = add_business_days(cursor, 1)
        steps.append(f"{_weekday_name(cursor)} {cursor.isoformat()} = business day {i + 1}")

    return [
        (
            "expected_settlement_date",
            {
                "value": expected.isoformat(),
                "computed_from": {
                    "oldest_unsettled_batch_captured_date": captured_raw,
                    "captured_date_weekday": _weekday_name(captured),
                    "settlement_cycle": cycle,
                    "business_days_added": n,
                },
                "method": (
                    f"{n} business day(s) after {captured_raw} ({_weekday_name(captured)}), "
                    f"skipping weekends: " + "; ".join(steps) + ". "
                    "Weekends excluded; no bank-holiday calendar is defined in this corpus."
                ),
            },
        ),
        (
            "business_days_since_batch_captured",
            {
                "value": elapsed,
                "computed_from": {
                    "oldest_unsettled_batch_captured_date": captured_raw,
                    "as_of": today.isoformat(),
                },
                "method": (
                    f"Business days strictly after {captured_raw} up to and including "
                    f"{today.isoformat()} ({_weekday_name(today)}), weekends excluded."
                ),
            },
        ),
        (
            "settlement_is_overdue",
            {
                "value": today > expected,
                "computed_from": {
                    "expected_settlement_date": expected.isoformat(),
                    "as_of": today.isoformat(),
                },
                "method": (
                    f"True only if {today.isoformat()} is strictly later than the expected "
                    f"settlement date {expected.isoformat()}. "
                    f"{today.isoformat()} is "
                    f"{'later than' if today > expected else 'not later than'} "
                    f"{expected.isoformat()}, so the settlement is "
                    f"{'overdue' if today > expected else 'not overdue'}."
                ),
            },
        ),
    ]


def _facts_for_last_payout(merchant_id: str, merchant: dict) -> list[tuple[str, Any]]:
    last_payout = merchant.get("settlement_schedule", {}).get("last_payout", {})
    raw = last_payout.get("released_date") or last_payout.get("date")
    if not raw:
        return []
    today = _corpus_today()
    d = _parse(raw)
    return [
        (
            "calendar_days_since_last_payout",
            {
                "value": (today - d).days,
                "computed_from": {"last_payout_date": raw, "as_of": today.isoformat()},
                "method": f"Calendar days from {raw} to {today.isoformat()}.",
            },
        )
    ]


def _facts_for_disputes(merchant_id: str, merchant: dict) -> list[tuple[str, Any]]:
    facts: list[tuple[str, Any]] = []
    today = _corpus_today()
    low, high = DISPUTE_REVIEW_WINDOW_DAYS

    for i, dispute in enumerate(merchant.get("disputes", [])):
        submitted_raw = dispute.get("evidence_submitted_date")
        if not submitted_raw:
            continue
        submitted = _parse(submitted_raw)
        elapsed = (today - submitted).days
        facts.append(
            (
                f"disputes[{i}].calendar_days_since_evidence_submitted",
                {
                    "value": elapsed,
                    "computed_from": {
                        "dispute_id": dispute.get("id"),
                        "evidence_submitted_date": submitted_raw,
                        "as_of": today.isoformat(),
                    },
                    "method": f"Calendar days from {submitted_raw} to {today.isoformat()}.",
                },
            )
        )
        facts.append(
            (
                f"disputes[{i}].within_bank_review_window",
                {
                    "value": elapsed <= high,
                    "computed_from": {
                        "dispute_id": dispute.get("id"),
                        "days_elapsed": elapsed,
                        "window_days": list(DISPUTE_REVIEW_WINDOW_DAYS),
                        "as_of": today.isoformat(),
                    },
                    "method": (
                        f"{elapsed} day(s) have elapsed since evidence was submitted. The "
                        f"issuing bank's typical decision window is {low}-{high} days "
                        f"(corpus/policy/chargebacks-disputes.md#c4). {elapsed} is "
                        f"{'within' if elapsed <= high else 'beyond'} the upper bound of "
                        f"{high} days, so the dispute is "
                        f"{'still inside' if elapsed <= high else 'past'} the normal window."
                    ),
                },
            )
        )
    return facts


def get_derived_facts(merchant_id: str) -> list[dict] | dict:
    """Every derived fact available for a merchant, one evidence envelope
    each.

    Returns the same not-found sentinel as the state tools for a
    merchant_id that doesn't resolve, and an empty list for a real
    merchant with nothing time-relative to compute (a real, valid answer -
    not an error).
    """
    merchant = _load_merchant(merchant_id)
    if merchant is None:
        return _not_found(merchant_id)

    pairs: list[tuple[str, Any]] = []
    pairs.extend(_facts_for_settlement(merchant_id, merchant))
    pairs.extend(_facts_for_last_payout(merchant_id, merchant))
    pairs.extend(_facts_for_disputes(merchant_id, merchant))

    return [
        make_evidence(f"derived:{merchant_id}.{name}", content, "derived")
        for name, content in pairs
    ]
