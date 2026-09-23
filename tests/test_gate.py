"""Tests src/gate.py - one test per rule, plus the ticket_081 scenario
the UNCORROBORATED+money_movement rule exists to defend against.

The gate is the only component in this project that is purely
deterministic, so unlike the retrieval/verifier evals these are exact
assertions, not measured rates.
"""

import pytest

from src.gate import (
    AUTO_RESOLVE,
    DRAFT_FOR_HUMAN,
    ESCALATE,
    REQUEST_IDENTIFICATION,
    RULE_CLAIM_STRIPPED,
    RULE_CLEAN_AUTO_RESOLVE,
    RULE_INSUFFICIENT_SUPPORT_RATIO,
    RULE_MALFORMED_FAIL_CLOSED,
    RULE_MISMATCH_DATA_ACCESS_RISK,
    RULE_ROOT_CAUSE_REJECTED,
    RULE_TOO_FEW_CLAIMS,
    RULE_UNCORROBORATED_DISCLOSES_ACCOUNT_DATA,
    RULE_UNIDENTIFIABLE_ASK,
    evaluate,
    evaluate_all,
)


def make_diagnosis(
    identity_status="CONFIRMED",
    risk_class="informational",
    n_claims=2,
    verification=True,
    root_verdict="SUPPORTED",
    claim_verdicts=("SUPPORTED", "SUPPORTED"),
    claim_evidence="state:merchant_1",
    root_evidence=None,
):
    """A diagnosis shaped like what src/agent.py actually returns."""
    d = {
        "category": "settlement_timing",
        "risk_class": risk_class,
        "claims": [
            {"text": f"claim {i}", "evidence": [claim_evidence]} for i in range(n_claims)
        ],
        "root_cause": {
            "text": "root",
            "evidence": [root_evidence if root_evidence is not None else claim_evidence],
        },
        "recommended_action": "auto_resolve",
        "identity_status": identity_status,
    }
    if verification:
        d["_verification"] = {
            "root_cause_verdict": {"verdict": root_verdict, "reason": "r"},
            "verdicts": [
                {"claim": "root", "is_root_cause": True, "verdict": root_verdict, "reason": "r"},
                *[
                    {"claim": f"claim {i}", "is_root_cause": False, "verdict": v, "reason": "r"}
                    for i, v in enumerate(claim_verdicts)
                ],
            ],
        }
    return d


# --- identity rules -------------------------------------------------------

def test_unidentifiable_asks_for_identification_not_escalation():
    d = evaluate(make_diagnosis(identity_status="UNIDENTIFIABLE"))
    assert d.outcome == REQUEST_IDENTIFICATION
    assert d.rule == RULE_UNIDENTIFIABLE_ASK
    assert d.outcome != ESCALATE  # the distinction is the point of the outcome


def test_unidentifiable_wins_even_on_an_otherwise_perfect_diagnosis():
    d = evaluate(make_diagnosis(identity_status="UNIDENTIFIABLE", risk_class="informational"))
    assert d.outcome == REQUEST_IDENTIFICATION


def test_mismatch_always_escalates_and_is_flagged_as_data_access_risk():
    d = evaluate(make_diagnosis(identity_status="MISMATCH"))
    assert d.outcome == ESCALATE
    assert d.rule == RULE_MISMATCH_DATA_ACCESS_RISK
    assert "data_access_risk" in d.risk_flags


@pytest.mark.parametrize("risk_class", ["money_movement", "informational"])
def test_mismatch_escalates_regardless_of_risk_class_or_verdicts(risk_class):
    d = evaluate(
        make_diagnosis(identity_status="MISMATCH", risk_class=risk_class, root_verdict="SUPPORTED")
    )
    assert d.outcome == ESCALATE


# --- the ticket_081 defence: disclosure, not category --------------------

def test_uncorroborated_identity_citing_account_data_never_auto_resolves():
    """ticket_081 claims merchant_3 but is actually merchant_12, and names
    no id and no business name - so identity resolution correctly returns
    UNCORROBORATED and cannot catch it (tests/test_identity.py documents
    this). The gate is the remaining defence, and it keys on what the
    answer would disclose: an answer quoting this account's own state to
    an unverified sender gets a human first.
    """
    d = evaluate(
        make_diagnosis(
            identity_status="UNCORROBORATED",
            risk_class="money_movement",
            claim_evidence="state:merchant_3.kyc.documents[1].rejection_reason",
            claim_verdicts=("SUPPORTED", "SUPPORTED"),
            root_verdict="SUPPORTED",
        )
    )
    assert d.outcome == DRAFT_FOR_HUMAN
    assert d.rule == RULE_UNCORROBORATED_DISCLOSES_ACCOUNT_DATA
    assert "account_data_disclosure" in d.risk_flags


def test_uncorroborated_informational_citing_state_must_not_auto_resolve():
    """The case the old money_movement proxy missed: an informational
    ticket (a webhook question) whose answer still quotes the merchant's
    webhook log and transaction ids. Nothing about money moves, but one
    merchant's account data would still reach an unverified sender.
    """
    d = evaluate(
        make_diagnosis(
            identity_status="UNCORROBORATED",
            risk_class="informational",
            claim_evidence="state:merchant_7.webhook_log[0]",
        )
    )
    assert d.outcome == DRAFT_FOR_HUMAN
    assert d.rule == RULE_UNCORROBORATED_DISCLOSES_ACCOUNT_DATA


def test_uncorroborated_policy_only_answer_may_auto_resolve():
    """The other side of the same rule: an answer built purely from
    published policy discloses nothing account-specific, so an
    uncorroborated identity is not a disclosure risk even if we have the
    wrong merchant.
    """
    d = evaluate(
        make_diagnosis(
            identity_status="UNCORROBORATED",
            risk_class="informational",
            claim_evidence="doc:refund-timelines#c2",
        )
    )
    assert d.outcome == AUTO_RESOLVE
    assert d.rule == RULE_CLEAN_AUTO_RESOLVE


def test_uncorroborated_policy_only_money_movement_may_auto_resolve():
    # deliberately unblocked by the change: "how long do card refunds
    # take" answered purely from policy is safe to send whoever asked,
    # even though the topic is money.
    d = evaluate(
        make_diagnosis(
            identity_status="UNCORROBORATED",
            risk_class="money_movement",
            claim_evidence="doc:refund-timelines#c2",
        )
    )
    assert d.outcome == AUTO_RESOLVE


def test_derived_evidence_counts_as_account_data_disclosure():
    # a derived fact is computed from this merchant's own state, so
    # "your batch was captured 2 business days ago" discloses account
    # activity just as the raw field would.
    d = evaluate(
        make_diagnosis(
            identity_status="UNCORROBORATED",
            risk_class="informational",
            claim_evidence="derived:merchant_1.expected_settlement_date",
        )
    )
    assert d.outcome == DRAFT_FOR_HUMAN
    assert d.rule == RULE_UNCORROBORATED_DISCLOSES_ACCOUNT_DATA


def test_root_cause_citing_state_triggers_disclosure_rule_even_if_claims_are_policy_only():
    # the root cause is rendered into the message and listed in Sources,
    # so it discloses too - the rule reads both.
    d = evaluate(
        make_diagnosis(
            identity_status="UNCORROBORATED",
            risk_class="informational",
            claim_evidence="doc:refund-timelines#c2",
            root_evidence="state:merchant_8.refunds[0]",
        )
    )
    assert d.outcome == DRAFT_FOR_HUMAN
    assert d.rule == RULE_UNCORROBORATED_DISCLOSES_ACCOUNT_DATA


def test_disclosure_rule_fires_even_with_perfect_verification():
    d = evaluate(
        make_diagnosis(
            identity_status="UNCORROBORATED",
            risk_class="money_movement",
            claim_evidence="state:merchant_1",
            n_claims=5,
            claim_verdicts=("SUPPORTED",) * 5,
        )
    )
    assert d.outcome == DRAFT_FOR_HUMAN


def test_corroborated_identity_citing_state_still_auto_resolves():
    # the rule is about UNCORROBORATED identity, not about citing state -
    # a confirmed merchant getting their own data back is the normal case.
    d = evaluate(
        make_diagnosis(identity_status="CONFIRMED", claim_evidence="state:merchant_1")
    )
    assert d.outcome == AUTO_RESOLVE


# --- verification rules ---------------------------------------------------

def test_rejected_root_cause_escalates():
    d = evaluate(make_diagnosis(root_verdict="UNSUPPORTED"))
    assert d.outcome == ESCALATE
    assert d.rule == RULE_ROOT_CAUSE_REJECTED
    assert "ungrounded_root_cause" in d.risk_flags


def test_a_stripped_claim_blocks_auto_resolve():
    d = evaluate(make_diagnosis(claim_verdicts=("SUPPORTED", "UNSUPPORTED")))
    assert d.outcome == DRAFT_FOR_HUMAN
    assert d.rule == RULE_CLAIM_STRIPPED


def test_partially_supported_claims_fall_below_the_ratio_bar():
    d = evaluate(make_diagnosis(claim_verdicts=("SUPPORTED", "PARTIALLY_SUPPORTED")))
    assert d.outcome == DRAFT_FOR_HUMAN
    assert d.rule == RULE_INSUFFICIENT_SUPPORT_RATIO


def test_all_supported_claims_auto_resolve():
    d = evaluate(make_diagnosis(claim_verdicts=("SUPPORTED", "SUPPORTED")))
    assert d.outcome == AUTO_RESOLVE
    assert d.rule == RULE_CLEAN_AUTO_RESOLVE


def test_zero_surviving_claims_blocks_auto_resolve():
    d = evaluate(make_diagnosis(n_claims=0, claim_verdicts=()))
    assert d.outcome == DRAFT_FOR_HUMAN
    assert d.rule == RULE_TOO_FEW_CLAIMS


# --- money_movement is a stricter bar than informational ------------------

def test_money_movement_bar_is_at_least_as_strict_as_informational():
    from src.config import AUTO_RESOLVE_MIN_CLAIMS, AUTO_RESOLVE_MIN_SUPPORTED_RATIO

    assert (
        AUTO_RESOLVE_MIN_SUPPORTED_RATIO["money_movement"]
        >= AUTO_RESOLVE_MIN_SUPPORTED_RATIO["informational"]
    )
    assert AUTO_RESOLVE_MIN_CLAIMS["money_movement"] >= AUTO_RESOLVE_MIN_CLAIMS["informational"]


def test_money_movement_bar_is_strictly_stricter_somewhere():
    """">= on both" is satisfied by two identical bars, which would mean
    "money_movement is stricter" was never actually implemented. At least
    one dimension must genuinely differ.
    """
    from src.config import AUTO_RESOLVE_MIN_CLAIMS, AUTO_RESOLVE_MIN_SUPPORTED_RATIO

    assert (
        AUTO_RESOLVE_MIN_SUPPORTED_RATIO["money_movement"]
        > AUTO_RESOLVE_MIN_SUPPORTED_RATIO["informational"]
        or AUTO_RESOLVE_MIN_CLAIMS["money_movement"] > AUTO_RESOLVE_MIN_CLAIMS["informational"]
    )


def test_a_single_claim_money_movement_answer_cannot_auto_resolve():
    # the concrete effect of the stricter bar: a settlement/refund answer
    # resting on one fact goes to a human even if that fact is supported.
    d = evaluate(
        make_diagnosis(
            identity_status="CONFIRMED",
            risk_class="money_movement",
            n_claims=1,
            claim_verdicts=("SUPPORTED",),
        )
    )
    assert d.outcome == DRAFT_FOR_HUMAN
    assert d.rule == RULE_TOO_FEW_CLAIMS


def test_a_single_claim_informational_answer_may_auto_resolve():
    d = evaluate(
        make_diagnosis(
            identity_status="CONFIRMED",
            risk_class="informational",
            n_claims=1,
            claim_verdicts=("SUPPORTED",),
        )
    )
    assert d.outcome == AUTO_RESOLVE


# --- verification absent (the ablation) ----------------------------------

def test_missing_verification_is_not_treated_as_failed_verification():
    """With the verifier off the gate simply has fewer signals. Treating
    absent verdicts as failures would route every ablation ticket to
    DRAFT_FOR_HUMAN and make the measured verifier delta an artifact of
    the gate rather than a property of the verifier.
    """
    d = evaluate(make_diagnosis(verification=False))
    assert d.outcome == AUTO_RESOLVE
    assert d.inputs["verification"]["ran"] is False


def test_identity_rules_still_apply_with_verification_off():
    assert evaluate(make_diagnosis(identity_status="MISMATCH", verification=False)).outcome == ESCALATE
    assert (
        evaluate(make_diagnosis(identity_status="UNIDENTIFIABLE", verification=False)).outcome
        == REQUEST_IDENTIFICATION
    )
    # the disclosure rule reads evidence ids, which exist with or without
    # verification - so it still fires here, and for the right reason.
    uncorroborated = evaluate(
        make_diagnosis(
            identity_status="UNCORROBORATED",
            claim_evidence="state:merchant_1",
            verification=False,
        )
    )
    assert uncorroborated.outcome == DRAFT_FOR_HUMAN
    assert uncorroborated.rule == RULE_UNCORROBORATED_DISCLOSES_ACCOUNT_DATA


# --- fail closed ----------------------------------------------------------

@pytest.mark.parametrize(
    "broken",
    [
        {},
        {"identity_status": "CONFIRMED"},  # no risk_class
        {"identity_status": "NOT_A_STATUS", "risk_class": "informational"},
        {"identity_status": "CONFIRMED", "risk_class": "not_a_class"},
        {"identity_status": None, "risk_class": None},
    ],
)
def test_malformed_diagnosis_escalates_rather_than_raising(broken):
    d = evaluate(broken)
    assert d.outcome == ESCALATE
    assert d.rule == RULE_MALFORMED_FAIL_CLOSED


def test_corrupt_verification_record_escalates():
    d = make_diagnosis()
    d["_verification"] = {"verdicts": "not-a-list"}
    assert evaluate(d).outcome == ESCALATE


# --- every decision carries its rule, for the audit trail ----------------

def test_every_decision_reports_the_rule_that_fired_and_its_inputs():
    for diagnosis in [
        make_diagnosis(identity_status="UNIDENTIFIABLE"),
        make_diagnosis(identity_status="MISMATCH"),
        make_diagnosis(root_verdict="UNSUPPORTED"),
        make_diagnosis(identity_status="UNCORROBORATED", risk_class="money_movement"),
        make_diagnosis(claim_verdicts=("SUPPORTED", "UNSUPPORTED")),
        make_diagnosis(),
    ]:
        d = evaluate(diagnosis)
        assert d.rule
        assert d.reason
        assert d.outcome in (AUTO_RESOLVE, DRAFT_FOR_HUMAN, REQUEST_IDENTIFICATION, ESCALATE)
        assert d.as_dict()["rule"] == d.rule


# --- multi-cause aggregation ---------------------------------------------

def test_multi_cause_takes_the_most_conservative_lane():
    clean = make_diagnosis()  # AUTO_RESOLVE
    rejected = make_diagnosis(root_verdict="UNSUPPORTED")  # ESCALATE
    d = evaluate_all([clean, rejected])
    assert d.outcome == ESCALATE
    assert "2 causes" in d.reason


def test_multi_cause_with_one_draft_and_one_auto_resolves_to_draft():
    clean = make_diagnosis()
    stripped = make_diagnosis(claim_verdicts=("SUPPORTED", "UNSUPPORTED"))
    assert evaluate_all([clean, stripped]).outcome == DRAFT_FOR_HUMAN


def test_multi_cause_all_clean_still_auto_resolves():
    assert evaluate_all([make_diagnosis(), make_diagnosis()]).outcome == AUTO_RESOLVE


def test_single_diagnosis_through_evaluate_all_is_unchanged():
    one = make_diagnosis()
    assert evaluate_all([one].copy()).as_dict() == evaluate(one).as_dict()


def test_no_diagnoses_at_all_escalates():
    d = evaluate_all([])
    assert d.outcome == ESCALATE
    assert d.rule == RULE_MALFORMED_FAIL_CLOSED
