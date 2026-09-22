import pytest

from src.evidence import InvalidEvidenceId, make_evidence


def test_envelope_shape():
    e = make_evidence("doc:settlement-cycles#c1", "some text", "policy")
    assert set(e.keys()) == {"evidence_id", "content", "source_type"}
    assert e["evidence_id"] == "doc:settlement-cycles#c1"
    assert e["content"] == "some text"
    assert e["source_type"] == "policy"


def test_strips_top_level_hidden_key():
    content = {"amount": 500, "_derivation": "the answer is X"}
    e = make_evidence("state:merchant_1.settlement_schedule", content, "state")
    assert "_derivation" not in e["content"]
    assert e["content"] == {"amount": 500}


def test_strips_nested_hidden_keys_in_dicts_and_lists():
    content = {
        "reserve": {
            "percentage": 15,
            "_derivation": "leak",
            "release_schedule": [
                {"amount": 100, "_derivation": "leak2"},
                {"amount": 200},
            ],
        }
    }
    e = make_evidence("state:merchant_2.reserve", content, "state")
    assert "_derivation" not in e["content"]["reserve"]
    for item in e["content"]["reserve"]["release_schedule"]:
        assert "_derivation" not in item
    # non-hidden data survives untouched
    assert e["content"]["reserve"]["percentage"] == 15
    assert e["content"]["reserve"]["release_schedule"][1] == {"amount": 200}


def test_does_not_mutate_input():
    original = {"amount": 500, "_derivation": "leak"}
    make_evidence("state:merchant_1.x", original, "state")
    assert "_derivation" in original, "make_evidence must not mutate its input"


def test_rejects_invalid_source_type():
    with pytest.raises(ValueError):
        make_evidence("doc:settlement-cycles#c1", "x", "web")


@pytest.mark.parametrize(
    "evidence_id",
    [
        "settlement-cycles#c1",  # missing doc: prefix
        "doc:settlement-cycles",  # missing #c<n>
        "doc:Settlement-Cycles#c1",  # uppercase slug
        "doc:settlement-cycles#c0",  # chunk numbers are 1-indexed
        "doc:settlement-cycles#3",  # missing 'c'
    ],
)
def test_rejects_malformed_policy_evidence_id(evidence_id):
    with pytest.raises(InvalidEvidenceId):
        make_evidence(evidence_id, "x", "policy")


@pytest.mark.parametrize(
    "evidence_id",
    [
        "state:merchant_1.settlement_schedule.next_payout",
        "state:merchant_3.transactions[12].status",
        "state:merchant_10.international_payments.approval_date",
        "state:merchant_1",  # bare merchant ref: cites the whole record
    ],
)
def test_accepts_valid_state_evidence_id(evidence_id):
    e = make_evidence(evidence_id, "x", "state")
    assert e["evidence_id"] == evidence_id


@pytest.mark.parametrize(
    "evidence_id",
    [
        "merchant_1.settlement_schedule",  # missing state: prefix
        "state:merchant_1.",  # trailing dot, no segment after it
        "state:merchant_x.settlement_schedule",  # merchant id not the merchant_<digits> form
        "state:merchant_1.transactions[abc]",  # non-numeric index
    ],
)
def test_rejects_malformed_state_evidence_id(evidence_id):
    with pytest.raises(InvalidEvidenceId):
        make_evidence(evidence_id, "x", "state")


# --- derived facts (src/derived_facts.py) --------------------------------

@pytest.mark.parametrize(
    "evidence_id",
    [
        "derived:merchant_1.expected_settlement_date",
        "derived:merchant_9.disputes[0].within_bank_review_window",
        "derived:merchant_13.calendar_days_since_last_payout",
    ],
)
def test_accepts_valid_derived_evidence_id(evidence_id):
    e = make_evidence(evidence_id, {"value": 1}, "derived")
    assert e["evidence_id"] == evidence_id
    assert e["source_type"] == "derived"


@pytest.mark.parametrize(
    "evidence_id",
    [
        "derived:merchant_1",  # bare merchant: no such thing as a whole derived record
        "merchant_1.expected_settlement_date",  # missing derived: prefix
        "derived:merchant_x.expected_settlement_date",  # bad merchant id form
        "state:merchant_1.expected_settlement_date",  # right shape, wrong prefix for this type
    ],
)
def test_rejects_malformed_derived_evidence_id(evidence_id):
    with pytest.raises(InvalidEvidenceId):
        make_evidence(evidence_id, {"value": 1}, "derived")


def test_derived_content_is_hidden_key_stripped_like_any_other_source():
    e = make_evidence(
        "derived:merchant_1.expected_settlement_date",
        {"value": "2026-09-18", "_derivation": "leak"},
        "derived",
    )
    assert "_derivation" not in e["content"]
