"""Tests src/derived_facts.py - above all the business-day arithmetic,
which exists precisely because the verifier cannot be trusted to do it
(it has no calendar and no notion of "now"). If this module is wrong, the
verifier has no way to notice, so these tests are the only thing standing
behind every time-relative claim the agent makes.
"""

from datetime import date

import pytest

from src.config import CORPUS_NOW
from src.derived_facts import (
    add_business_days,
    business_days_between,
    get_derived_facts,
    is_business_day,
)
from src.state_tools import is_not_found

# 2026-09-14 is a Monday; the week runs Mon 14 ... Sun 20.
MON = date(2026, 9, 14)
TUE = date(2026, 9, 15)
WED = date(2026, 9, 16)
THU = date(2026, 9, 17)
FRI = date(2026, 9, 18)
SAT = date(2026, 9, 19)
SUN = date(2026, 9, 20)
NEXT_MON = date(2026, 9, 21)


def test_weekday_calendar_assumption_holds():
    # every other test in this file depends on these being the real
    # weekdays - if the calendar assumption is wrong, everything below is
    # meaningless rather than merely failing.
    assert MON.strftime("%A") == "Monday"
    assert SAT.strftime("%A") == "Saturday"
    assert SUN.strftime("%A") == "Sunday"


# --- is_business_day ------------------------------------------------------

@pytest.mark.parametrize("d", [MON, TUE, WED, THU, FRI])
def test_weekdays_are_business_days(d):
    assert is_business_day(d) is True


@pytest.mark.parametrize("d", [SAT, SUN])
def test_weekend_days_are_not_business_days(d):
    assert is_business_day(d) is False


# --- add_business_days ---------------------------------------------------

def test_t_plus_3_from_tuesday_lands_on_friday():
    # the exact case adv_1 got wrong: Tue + 3 business days = Fri, not Thu.
    assert add_business_days(TUE, 3) == FRI


def test_t_plus_3_from_wednesday_skips_the_weekend_to_monday():
    assert add_business_days(WED, 3) == NEXT_MON


def test_t_plus_1_from_friday_skips_the_weekend():
    assert add_business_days(FRI, 1) == NEXT_MON


def test_counting_from_a_saturday_starts_at_the_next_business_day():
    assert add_business_days(SAT, 1) == NEXT_MON


def test_counting_from_a_sunday_starts_at_the_next_business_day():
    assert add_business_days(SUN, 1) == NEXT_MON


def test_zero_business_days_is_the_start_date_itself():
    assert add_business_days(TUE, 0) == TUE


def test_spanning_multiple_weekends():
    # 10 business days from Mon 2026-09-14 = Mon 2026-09-28 (two weekends).
    assert add_business_days(MON, 10) == date(2026, 9, 28)


def test_negative_days_rejected():
    with pytest.raises(ValueError):
        add_business_days(TUE, -1)


# --- business_days_between -----------------------------------------------

def test_business_days_between_is_the_inverse_of_add():
    for n in range(0, 15):
        assert business_days_between(TUE, add_business_days(TUE, n)) == n


def test_business_days_between_excludes_the_weekend():
    # Fri -> next Mon is one business day, not three calendar days.
    assert business_days_between(FRI, NEXT_MON) == 1


def test_business_days_between_same_day_is_zero():
    assert business_days_between(TUE, TUE) == 0


def test_business_days_between_backwards_is_zero_not_negative():
    assert business_days_between(FRI, TUE) == 0


# --- derived facts over the real corpus ----------------------------------

def test_expected_settlement_date_matches_the_fixtures_own_next_payout():
    """The strongest available check that the arithmetic is right: the
    derived value is computed from captured_date + cycle alone, with no
    reference to next_payout.date, yet must agree with it.
    """
    from src.state_tools import get_settlement_schedule

    for merchant_id in ("merchant_1", "merchant_3", "merchant_13"):
        facts = {f["evidence_id"]: f["content"] for f in get_derived_facts(merchant_id)}
        derived = facts[f"derived:{merchant_id}.expected_settlement_date"]["value"]
        fixture = get_settlement_schedule(merchant_id)["content"]["next_payout"]["date"]
        assert derived == fixture, f"{merchant_id}: derived {derived} != fixture {fixture}"


def test_merchant_1_expected_settlement_is_the_18th_not_the_17th():
    facts = {f["evidence_id"]: f["content"] for f in get_derived_facts("merchant_1")}
    assert facts["derived:merchant_1.expected_settlement_date"]["value"] == "2026-09-18"
    assert facts["derived:merchant_1.settlement_is_overdue"]["value"] is False


def test_dispute_window_check_computes_elapsed_days_and_the_boolean():
    facts = {f["evidence_id"]: f["content"] for f in get_derived_facts("merchant_9")}
    elapsed = facts["derived:merchant_9.disputes[0].calendar_days_since_evidence_submitted"]
    within = facts["derived:merchant_9.disputes[0].within_bank_review_window"]
    assert elapsed["value"] == 33  # 2026-08-15 -> 2026-09-17
    assert within["value"] is True
    assert within["computed_from"]["window_days"] == [30, 45]


def test_every_derived_fact_is_a_valid_evidence_envelope():
    for merchant_id in ("merchant_1", "merchant_9", "merchant_14"):
        for fact in get_derived_facts(merchant_id):
            assert set(fact.keys()) == {"evidence_id", "content", "source_type"}
            assert fact["source_type"] == "derived"
            assert fact["evidence_id"].startswith(f"derived:{merchant_id}.")


def test_time_relative_facts_carry_the_as_of_date_they_used():
    """A verifier sees only the content, so anything computed relative to
    now must say which "now" - otherwise the claim is uncheckable for the
    same reason adv_1 was.
    """
    facts = {f["evidence_id"]: f["content"] for f in get_derived_facts("merchant_1")}
    for key in (
        "derived:merchant_1.business_days_since_batch_captured",
        "derived:merchant_1.settlement_is_overdue",
        "derived:merchant_1.calendar_days_since_last_payout",
    ):
        assert facts[key]["computed_from"]["as_of"] == CORPUS_NOW.date().isoformat()


def test_nonexistent_merchant_returns_not_found_sentinel():
    assert is_not_found(get_derived_facts("merchant_999"))


def test_derived_facts_never_leak_hidden_keys():
    for merchant_id in ("merchant_1", "merchant_2", "merchant_9"):
        for fact in get_derived_facts(merchant_id):
            assert "_derivation" not in repr(fact)
