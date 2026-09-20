"""Tests the merchant identification pipeline against the exact tickets
it exists to get right: the four identity-mismatch training tickets
(017, 081, 082, 083) and the six missing-merchant_id tickets, plus a
handful of ordinary CONFIRMED cases so the pipeline isn't only ever
tested on its hard cases.
"""

import json

import pytest

from src.identity import NAME_MATCH_THRESHOLD, identify_merchant

TICKETS_DIR = "corpus/tickets"


def _load(ticket_id: str) -> dict:
    return json.load(open(f"{TICKETS_DIR}/{ticket_id}.json"))


# --- the four identity-mismatch training tickets ------------------------

def test_ticket_017_obvious_mismatch_caught_via_dispute_reference():
    result = identify_merchant(_load("ticket_017"))
    assert result["outcome"] == "MISMATCH"
    assert result["identified_merchant_id"] == "merchant_14"
    assert result["claimed_merchant_id"] == "merchant_9"
    assert "merchant_14" in result["signals"]["reference_matches"]


def test_ticket_082_obvious_mismatch_caught_via_dispute_reference():
    result = identify_merchant(_load("ticket_082"))
    assert result["outcome"] == "MISMATCH"
    assert result["identified_merchant_id"] == "merchant_14"
    assert result["claimed_merchant_id"] == "merchant_5"


def test_ticket_081_plausible_trap_is_not_caught_and_that_is_the_honest_result():
    """ticket_081 claims merchant_3 but is actually merchant_12 - the
    training corpus's "plausible-match trap" (both accounts are, right
    now, genuinely KYC-rejected with a same-file resubmission, so a wrong
    answer here is confidently wrong, not an obvious miss).

    This test documents, on purpose, that resolve_reference/
    find_merchant_by_name CANNOT catch it: the ticket body names no id and
    no business name, only a business *description* ("tiffin delivery
    service"). The correct behavior for an identification-only step is
    UNCORROBORATED, not a lucky MISMATCH - catching this specific trap
    requires comparing the claimed merchant's actual KYC rejection reason
    against what the ticket describes, which is diagnostic reasoning, not
    identification, and belongs to the agent loop (Unit 2), not this
    module.
    """
    result = identify_merchant(_load("ticket_081"))
    assert result["outcome"] == "UNCORROBORATED"
    assert result["identified_merchant_id"] == "merchant_3"  # the (wrong) claim, uncaught
    assert result["signals"]["references"] == []
    assert result["signals"]["names"] == []


def test_ticket_083_invalid_claimed_id_is_unidentifiable():
    result = identify_merchant(_load("ticket_083"))
    assert result["outcome"] == "UNIDENTIFIABLE"
    assert result["identified_merchant_id"] is None
    assert result["claimed_merchant_id"] == "merchant_23"


# --- the six missing-merchant_id tickets --------------------------------

@pytest.mark.parametrize(
    "ticket_id",
    ["ticket_016", "ticket_056", "ticket_077", "ticket_078", "ticket_079", "ticket_080"],
)
def test_missing_merchant_id_tickets_are_unidentifiable(ticket_id):
    ticket = _load(ticket_id)
    assert ticket["merchant_id"] is None  # sanity check on the fixture itself
    result = identify_merchant(ticket)
    assert result["outcome"] == "UNIDENTIFIABLE"
    assert result["identified_merchant_id"] is None


# --- ordinary CONFIRMED cases --------------------------------------------

def test_confirmed_via_customer_ref_reference():
    result = identify_merchant(_load("ticket_006"))  # cites cust_88213, claims merchant_6
    assert result["outcome"] == "CONFIRMED"
    assert result["identified_merchant_id"] == "merchant_6"


def test_confirmed_via_business_name_cue():
    result = identify_merchant(_load("ticket_002"))  # "I run Trailblazer Holiday Co."
    assert result["outcome"] == "CONFIRMED"
    assert result["identified_merchant_id"] == "merchant_2"
    assert result["signals"]["name_matches"] == ["merchant_2"]


def test_confirmed_via_dispute_reference_agreeing_with_claim():
    result = identify_merchant(_load("ticket_014"))  # cites disp_14_01, claims merchant_14
    assert result["outcome"] == "CONFIRMED"
    assert result["identified_merchant_id"] == "merchant_14"


def test_confirmed_with_no_claimed_id_but_a_reference_present():
    ticket = {"merchant_id": None, "body": "any update on disp_9_01?"}
    result = identify_merchant(ticket)
    assert result["outcome"] == "CONFIRMED"
    assert result["identified_merchant_id"] == "merchant_9"


# --- name-match confidence threshold behavior ---------------------------

def test_low_confidence_name_candidate_does_not_corroborate():
    # a short, ambiguous title-case phrase that doesn't clear the
    # threshold against any real business name must not count as
    # corroboration - the claim falls through to UNCORROBORATED instead.
    ticket = {"merchant_id": "merchant_1", "body": "This is Some Random Words here."}
    result = identify_merchant(ticket)
    assert result["outcome"] in ("UNCORROBORATED", "CONFIRMED")
    if result["signals"]["names"]:
        for name in result["signals"]["names"]:
            from src.state_tools import find_merchant_by_name

            top = find_merchant_by_name(name, top_n=1)[0]
            if top["content"]["score"] >= NAME_MATCH_THRESHOLD:
                pytest.skip("candidate text happened to clear the threshold; not a useful case")
    assert result["outcome"] == "UNCORROBORATED"


# --- reference vs. name priority, and genuine internal contradiction ----

def test_name_signal_is_skipped_once_a_reference_resolves():
    # a coincidentally-present capitalized phrase must not be consulted at
    # all once a reference has already resolved - not just outvoted.
    ticket = {
        "merchant_id": "merchant_14",
        "body": "This is Random Nonsense Corp but the real case is disp_14_01.",
    }
    result = identify_merchant(ticket)
    assert result["outcome"] == "CONFIRMED"
    assert result["signals"]["name_matches"] == []  # never attempted


def test_two_references_resolving_to_different_merchants_is_unidentifiable():
    """A genuine internal contradiction: two references in the SAME
    ticket body resolve to two DIFFERENT merchants (disp_14_01 belongs to
    merchant_14, txn_2_001 belongs to merchant_2). No ticket in the
    training corpus happens to do this, so this is a synthetic case -
    included because silently picking one of two disagreeing references
    would be exactly the kind of unearned confidence this module exists
    to prevent, and that branch deserves its own real test rather than
    being incidentally implied by another one.
    """
    ticket = {
        "merchant_id": "merchant_1",
        "body": "Following up on disp_14_01 and also txn_2_001, both are ours.",
    }
    result = identify_merchant(ticket)
    assert result["signals"]["reference_matches"] == ["merchant_14", "merchant_2"]
    assert result["outcome"] == "UNIDENTIFIABLE"
    assert result["identified_merchant_id"] is None
