import pytest

from src.schema import DIAGNOSIS_JSON_SCHEMA, UngroundedClaimError, render_prose, validate_diagnosis


def test_risk_class_schema_defines_the_money_movement_taxonomy():
    # a live run against ticket_001 (settlement timing) came back
    # risk_class="informational" before this description existed - the
    # model had never been told settlements count as money_movement.
    description = DIAGNOSIS_JSON_SCHEMA["properties"]["risk_class"]["description"].lower()
    for term in ("settlement", "refund", "reserve", "hold"):
        assert term in description

VALID = {
    "category": "settlement_timing",
    "risk_class": "money_movement",
    "claims": [{"text": "Settlement is on schedule.", "evidence": ["state:merchant_1.settlement_schedule"]}],
    "root_cause": "T+3 cycle, within window.",
    "recommended_action": "auto_resolve",
}


def test_valid_diagnosis_passes():
    validate_diagnosis(VALID, {"state:merchant_1.settlement_schedule"})


def test_rejects_invalid_risk_class():
    bad = {**VALID, "risk_class": "not_a_real_class"}
    with pytest.raises(ValueError):
        validate_diagnosis(bad, {"state:merchant_1.settlement_schedule"})


def test_rejects_invalid_recommended_action():
    bad = {**VALID, "recommended_action": "shrug"}
    with pytest.raises(ValueError):
        validate_diagnosis(bad, {"state:merchant_1.settlement_schedule"})


def test_rejects_evidence_id_never_gathered():
    with pytest.raises(UngroundedClaimError):
        validate_diagnosis(VALID, known_evidence_ids=set())


def test_accepts_multiple_claims_each_grounded():
    diagnosis = {
        **VALID,
        "claims": [
            {"text": "Claim one.", "evidence": ["state:merchant_1.a"]},
            {"text": "Claim two.", "evidence": ["doc:settlement-cycles#c1"]},
        ],
    }
    validate_diagnosis(diagnosis, {"state:merchant_1.a", "doc:settlement-cycles#c1"})


def test_one_ungrounded_claim_among_several_still_fails():
    diagnosis = {
        **VALID,
        "claims": [
            {"text": "Claim one.", "evidence": ["state:merchant_1.a"]},
            {"text": "Claim two.", "evidence": ["state:merchant_1.hallucinated"]},
        ],
    }
    with pytest.raises(UngroundedClaimError):
        validate_diagnosis(diagnosis, {"state:merchant_1.a"})


def test_render_prose_joins_claim_text_and_lists_citations():
    diagnosis = {
        "claims": [
            {"text": "First fact.", "evidence": ["state:merchant_1.a"]},
            {"text": "Second fact.", "evidence": ["doc:settlement-cycles#c1"]},
        ]
    }
    prose = render_prose(diagnosis)
    assert "First fact." in prose
    assert "Second fact." in prose
    assert "state:merchant_1.a" in prose
    assert "doc:settlement-cycles#c1" in prose


def test_render_prose_with_no_claims_produces_empty_body_no_crash():
    assert render_prose({"claims": []}) == ""
