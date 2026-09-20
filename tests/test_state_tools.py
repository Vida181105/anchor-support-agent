import glob
import json

import pytest

from src.state_tools import (
    find_merchant_by_name,
    get_disputes,
    get_merchant_state,
    get_settlement_schedule,
    get_transactions,
    is_not_found,
)


def _find_hidden_keys(obj, path=""):
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k.startswith("_"):
                hits.append(p)
            hits.extend(_find_hidden_keys(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(_find_hidden_keys(v, f"{path}[{i}]"))
    return hits


ALL_MERCHANT_IDS = [
    json.load(open(f))["merchant_id"]
    for f in sorted(glob.glob("corpus/merchants/merchant_*.json"))
]


# --- not-found behavior -----------------------------------------------

def test_get_merchant_state_not_found_for_nonexistent_id():
    result = get_merchant_state("merchant_23")
    assert is_not_found(result)
    assert "evidence_id" not in result
    assert result["merchant_id"] == "merchant_23"


def test_get_settlement_schedule_not_found_for_nonexistent_id():
    assert is_not_found(get_settlement_schedule("merchant_999"))


def test_get_disputes_not_found_for_nonexistent_id():
    assert is_not_found(get_disputes("merchant_999"))


def test_get_transactions_not_found_for_nonexistent_id():
    assert is_not_found(get_transactions("merchant_999"))


def test_get_disputes_empty_list_is_not_the_not_found_sentinel():
    # merchant_1 exists and has zero disputes - that's a real, valid,
    # empty result, not a lookup failure.
    result = get_disputes("merchant_1")
    assert result == []
    assert not is_not_found(result)


# --- get_merchant_state --------------------------------------------------

def test_get_merchant_state_returns_valid_envelope():
    result = get_merchant_state("merchant_1")
    assert result["evidence_id"] == "state:merchant_1"
    assert result["source_type"] == "state"
    assert result["content"]["business_name"] == "Kavya Handloom Exports"


@pytest.mark.parametrize("merchant_id", ALL_MERCHANT_IDS)
def test_get_merchant_state_never_leaks_hidden_keys(merchant_id):
    result = get_merchant_state(merchant_id)
    assert _find_hidden_keys(result["content"]) == []


# --- get_settlement_schedule ----------------------------------------------

def test_get_settlement_schedule_returns_correct_evidence_id_and_content():
    result = get_settlement_schedule("merchant_1")
    assert result["evidence_id"] == "state:merchant_1.settlement_schedule"
    assert result["content"]["cycle"] == "T+3"


@pytest.mark.parametrize("merchant_id", ALL_MERCHANT_IDS)
def test_get_settlement_schedule_never_leaks_hidden_keys(merchant_id):
    result = get_settlement_schedule(merchant_id)
    assert _find_hidden_keys(result["content"]) == []


# --- get_disputes ----------------------------------------------------------

def test_get_disputes_returns_one_envelope_per_dispute():
    result = get_disputes("merchant_2")  # has 2 disputes in the fixture
    assert len(result) == 2
    assert result[0]["evidence_id"] == "state:merchant_2.disputes[0]"
    assert result[1]["evidence_id"] == "state:merchant_2.disputes[1]"
    assert all(r["source_type"] == "state" for r in result)


@pytest.mark.parametrize("merchant_id", ALL_MERCHANT_IDS)
def test_get_disputes_never_leaks_hidden_keys(merchant_id):
    for envelope in get_disputes(merchant_id):
        assert _find_hidden_keys(envelope["content"]) == []


# --- get_transactions --------------------------------------------------

def test_get_transactions_no_filter_returns_everything():
    all_txns = get_transactions("merchant_1", filters=None)
    with open("corpus/merchants/merchant_1.json") as f:
        raw = json.load(f)
    assert len(all_txns) == len(raw["transactions"])


def test_get_transactions_filters_by_status():
    failed = get_transactions("merchant_1", filters={"status": "failed"})
    assert failed  # merchant_1 has some failed transactions
    assert all(e["content"]["status"] == "failed" for e in failed)


def test_get_transactions_filters_by_method_and_failure_code():
    results = get_transactions(
        "merchant_10", filters={"status": "failed", "failure_code": "RISK_BLOCK"}
    )
    assert results
    assert all(e["content"]["failure_code"] == "RISK_BLOCK" for e in results)


def test_get_transactions_filters_by_date_range():
    results = get_transactions(
        "merchant_1", filters={"start_date": "2026-09-01", "end_date": "2026-09-30"}
    )
    assert results
    assert all("2026-09-01" <= e["content"]["date"] <= "2026-09-30" for e in results)


def test_get_transactions_evidence_id_indexes_original_unfiltered_list():
    with open("corpus/merchants/merchant_1.json") as f:
        raw = json.load(f)
    failed = get_transactions("merchant_1", filters={"status": "failed"})
    for envelope in failed:
        # evidence_id like state:merchant_1.transactions[N] must point back
        # to the SAME transaction in the raw, unfiltered fixture list.
        idx = int(envelope["evidence_id"].split("[")[1].rstrip("]"))
        assert raw["transactions"][idx]["id"] == envelope["content"]["id"]


@pytest.mark.parametrize("merchant_id", ALL_MERCHANT_IDS)
def test_get_transactions_never_leaks_hidden_keys(merchant_id):
    for envelope in get_transactions(merchant_id):
        assert _find_hidden_keys(envelope["content"]) == []


# --- find_merchant_by_name --------------------------------------------

def test_find_merchant_by_name_exact_match_ranks_first():
    results = find_merchant_by_name("Zenith Fitness Equipment")
    assert results[0]["content"]["merchant_id"] == "merchant_14"
    assert results[0]["content"]["score"] == 1.0


def test_find_merchant_by_name_partial_name_still_ranks_correct_merchant_first():
    # this is the exact shape of ticket_017/082's identity-mismatch clue:
    # a casual short mention, not the full registered name.
    results = find_merchant_by_name("Zenith")
    assert results[0]["content"]["merchant_id"] == "merchant_14"
    assert results[0]["content"]["score"] > 0.5


def test_find_merchant_by_name_returns_evidence_envelopes_sorted_descending():
    results = find_merchant_by_name("Trailblazer")
    scores = [r["content"]["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(r["source_type"] == "state" for r in results)


def test_find_merchant_by_name_respects_top_n():
    results = find_merchant_by_name("anything", top_n=3)
    assert len(results) == 3


def test_find_merchant_by_name_never_returns_not_found_sentinel():
    # even a nonsense query returns low-scoring candidates, not an error
    results = find_merchant_by_name("zzzzzznonexistentbusinesszzzzz")
    assert not is_not_found(results)
    assert isinstance(results, list)
