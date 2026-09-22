"""Tests src/verifier.py: verdict handling, fail-closed behavior on every
failure mode named in the spec (error, malformed output, invalid enum),
the root_cause-rejects-everything rule, and - critically - that the
verifier's prompt contains ONLY the claim text and its cited evidence,
never anything else it could use to infer the "intended" answer.
"""

import json

import pytest

from src.verifier import verify_claim, verify_diagnosis


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_turn(self, turns, model=None, tools=None, response_json_schema=None, temperature=0.0):
        self.calls.append(turns)
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self.script.pop(0)


def _verdict_response(verdict, reason="because"):
    return {"text": json.dumps({"verdict": verdict, "reason": reason})}


# --- verify_claim: normal verdicts ----------------------------------------

def test_supported_verdict_passes_through():
    llm = ScriptedLLM([_verdict_response("SUPPORTED", "matches exactly")])
    claim = {"text": "Settlement is T+3.", "evidence": ["state:merchant_1.settlement_schedule"]}
    evidence_content = {"state:merchant_1.settlement_schedule": {"cycle": "T+3"}}

    result = verify_claim(claim, evidence_content, llm)

    assert result == {"verdict": "SUPPORTED", "reason": "matches exactly"}


def test_partially_supported_verdict_passes_through():
    llm = ScriptedLLM([_verdict_response("PARTIALLY_SUPPORTED", "date is off by one day")])
    claim = {"text": "x", "evidence": ["e1"]}
    result = verify_claim(claim, {"e1": {"a": 1}}, llm)
    assert result["verdict"] == "PARTIALLY_SUPPORTED"


def test_unsupported_verdict_passes_through():
    llm = ScriptedLLM([_verdict_response("UNSUPPORTED", "contradicts the evidence")])
    claim = {"text": "x", "evidence": ["e1"]}
    result = verify_claim(claim, {"e1": {"a": 1}}, llm)
    assert result["verdict"] == "UNSUPPORTED"


# --- fail-closed on every named failure mode -------------------------------

def test_claim_with_no_evidence_is_unsupported_without_calling_the_model():
    llm = ScriptedLLM([])  # would raise if called
    claim = {"text": "x", "evidence": []}
    result = verify_claim(claim, {}, llm)
    assert result["verdict"] == "UNSUPPORTED"
    assert llm.calls == []


def test_missing_evidence_content_is_unsupported_without_calling_the_model():
    llm = ScriptedLLM([])
    claim = {"text": "x", "evidence": ["e1", "e2"]}
    result = verify_claim(claim, {"e1": {"a": 1}}, llm)  # e2 missing
    assert result["verdict"] == "UNSUPPORTED"
    assert "e2" in result["reason"]
    assert llm.calls == []


def test_verifier_exception_fails_closed_to_unsupported():
    class ExplodingLLM:
        def generate_turn(self, *a, **kw):
            raise RuntimeError("simulated network failure")

    claim = {"text": "x", "evidence": ["e1"]}
    result = verify_claim(claim, {"e1": {"a": 1}}, ExplodingLLM())
    assert result["verdict"] == "UNSUPPORTED"
    assert "simulated network failure" in result["reason"]


def test_malformed_json_output_fails_closed_to_unsupported():
    llm = ScriptedLLM([{"text": "not valid json{{{"}])
    claim = {"text": "x", "evidence": ["e1"]}
    result = verify_claim(claim, {"e1": {"a": 1}}, llm)
    assert result["verdict"] == "UNSUPPORTED"


def test_invalid_verdict_enum_fails_closed_to_unsupported():
    llm = ScriptedLLM([{"text": json.dumps({"verdict": "MOSTLY_FINE_PROBABLY", "reason": "x"})}])
    claim = {"text": "x", "evidence": ["e1"]}
    result = verify_claim(claim, {"e1": {"a": 1}}, llm)
    assert result["verdict"] == "UNSUPPORTED"


# --- the verifier sees ONLY claim text + its cited evidence content -------

def test_prompt_contains_only_claim_text_and_cited_evidence_nothing_else():
    llm = ScriptedLLM([_verdict_response("SUPPORTED")])
    claim = {"text": "The refund was already processed.", "evidence": ["state:merchant_7.refunds[0]"]}
    evidence_content = {
        "state:merchant_7.refunds[0]": {"status": "processed", "amount": 5900},
        # extra entries NOT cited by this claim - must never reach the prompt
        "state:merchant_7.settlement_schedule": {"cycle": "T+3"},
        "doc:webhooks#c6": "unrelated policy text",
    }

    verify_claim(claim, evidence_content, llm)

    sent_text = llm.calls[0][0]["text"]
    assert "The refund was already processed." in sent_text
    assert "processed" in sent_text and "5900" in sent_text
    # the uncited evidence must not appear at all
    assert "T+3" not in sent_text
    assert "unrelated policy text" not in sent_text


def test_prompt_never_contains_ticket_or_merchant_fields():
    # even if a caller accidentally includes ticket/merchant context inside
    # a claim dict's extra keys, verify_claim must not read or forward
    # anything beyond "text" and "evidence".
    llm = ScriptedLLM([_verdict_response("SUPPORTED")])
    claim = {
        "text": "Settlement is on schedule.",
        "evidence": ["e1"],
        "ticket_body": "sir my payment not come till now 2 days already gone",
        "merchant_id": "merchant_1",
    }
    verify_claim(claim, {"e1": {"cycle": "T+3"}}, llm)

    sent_text = llm.calls[0][0]["text"]
    assert "payment not come" not in sent_text
    assert "merchant_1" not in sent_text


# --- verify_diagnosis: root_cause rejection rule ---------------------------

def test_unsupported_root_cause_rejects_the_entire_diagnosis():
    llm = ScriptedLLM([_verdict_response("UNSUPPORTED", "root cause overstates the evidence")])
    diagnosis = {
        "category": "settlement_timing",
        "risk_class": "money_movement",
        "claims": [{"text": "Some claim.", "evidence": ["e1"]}],
        "root_cause": {"text": "Root cause text.", "evidence": ["e1"]},
        "recommended_action": "auto_resolve",
        "identity_status": "CONFIRMED",
    }
    evidence_content = {"e1": {"cycle": "T+3"}}

    result = verify_diagnosis(diagnosis, evidence_content, llm)

    assert result["claims"] == []
    assert result["recommended_action"] == "escalate"
    assert "root cause overstates the evidence" in result["root_cause"]["text"]
    assert result["identity_status"] == "CONFIRMED"  # preserved through rejection
    # the claim's own verdict was never even checked - root_cause fails first
    assert len(llm.calls) == 1


def test_supported_root_cause_lets_claim_verification_proceed():
    llm = ScriptedLLM(
        [
            _verdict_response("SUPPORTED"),  # root_cause
            _verdict_response("SUPPORTED"),  # claim 1
            _verdict_response("UNSUPPORTED", "not in the evidence"),  # claim 2
        ]
    )
    diagnosis = {
        "category": "x",
        "risk_class": "informational",
        "claims": [
            {"text": "Claim one.", "evidence": ["e1"]},
            {"text": "Claim two.", "evidence": ["e2"]},
        ],
        "root_cause": {"text": "Root cause.", "evidence": ["e1"]},
        "recommended_action": "auto_resolve",
    }
    evidence_content = {"e1": {"a": 1}, "e2": {"b": 2}}

    result = verify_diagnosis(diagnosis, evidence_content, llm)

    assert [c["text"] for c in result["claims"]] == ["Claim one."]
    assert result["recommended_action"] == "auto_resolve"  # untouched - only root_cause rejection changes routing
    assert result["_verification"]["root_cause_verdict"]["verdict"] == "SUPPORTED"
    assert len(result["_verification"]["verdicts"]) == 3


def test_partially_supported_claims_are_kept_not_stripped():
    llm = ScriptedLLM(
        [
            _verdict_response("SUPPORTED"),  # root_cause
            _verdict_response("PARTIALLY_SUPPORTED", "date slightly off"),  # claim 1
        ]
    )
    diagnosis = {
        "category": "x",
        "risk_class": "informational",
        "claims": [{"text": "Claim one.", "evidence": ["e1"]}],
        "root_cause": {"text": "Root cause.", "evidence": ["e1"]},
        "recommended_action": "auto_resolve",
    }
    result = verify_diagnosis(diagnosis, {"e1": {"a": 1}}, llm)

    assert len(result["claims"]) == 1  # kept, not stripped
    assert result["_verification"]["verdicts"][1]["verdict"] == "PARTIALLY_SUPPORTED"


def test_all_claims_unsupported_leaves_empty_claims_list_but_keeps_root_cause():
    llm = ScriptedLLM(
        [
            _verdict_response("SUPPORTED"),  # root_cause
            _verdict_response("UNSUPPORTED"),  # only claim
        ]
    )
    diagnosis = {
        "category": "x",
        "risk_class": "informational",
        "claims": [{"text": "Claim.", "evidence": ["e1"]}],
        "root_cause": {"text": "Root cause.", "evidence": ["e1"]},
        "recommended_action": "auto_resolve",
    }
    result = verify_diagnosis(diagnosis, {"e1": {"a": 1}}, llm)
    assert result["claims"] == []
    assert result["root_cause"]["text"] == "Root cause."  # not rejected - root_cause itself was SUPPORTED
    assert result["recommended_action"] == "auto_resolve"
