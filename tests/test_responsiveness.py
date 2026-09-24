"""Tests src/responsiveness.py: verdict handling, fail-closed behaviour on
every failure mode (error, malformed output, invalid enum), and - the
point of the module - that the prompt contains the ticket text and the
root cause and NOTHING that would let it judge truth instead of relevance.

The mirror-image assertion lives in tests/test_verifier.py, which checks
the verifier never sees the ticket. Together they pin the property the
two-check design rests on: neither check has the information that would
let it rubber-stamp the other's job.
"""

import json

from src.responsiveness import check_diagnosis_responsiveness, check_responsiveness


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_turn(self, turns, model=None, tools=None, response_json_schema=None, temperature=0.0):
        self.calls.append(turns)
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self.script.pop(0)


class ExplodingLLM:
    def generate_turn(self, *a, **k):
        raise RuntimeError("503 UNAVAILABLE")


def _resp(verdict, reason="because"):
    return {"text": json.dumps({"verdict": verdict, "reason": reason})}


TICKET = "Hi, how do I change the bank account my settlements are paid into?"
ROOT_CAUSE = "Your settlement cycle is T+3 business days."


# --- normal verdicts -------------------------------------------------------

def test_responsive_verdict_passes_through():
    llm = ScriptedLLM([_resp("RESPONSIVE", "answers the question asked")])
    assert check_responsiveness(TICKET, "Here is how to change the account.", llm) == {
        "verdict": "RESPONSIVE",
        "reason": "answers the question asked",
    }


def test_partially_responsive_verdict_passes_through():
    llm = ScriptedLLM([_resp("PARTIALLY_RESPONSIVE", "only half")])
    assert check_responsiveness(TICKET, ROOT_CAUSE, llm)["verdict"] == "PARTIALLY_RESPONSIVE"


def test_non_responsive_verdict_passes_through():
    llm = ScriptedLLM([_resp("NON_RESPONSIVE", "different question")])
    assert check_responsiveness(TICKET, ROOT_CAUSE, llm)["verdict"] == "NON_RESPONSIVE"


# --- fail closed on every failure mode -------------------------------------

def test_api_error_fails_closed_to_non_responsive():
    result = check_responsiveness(TICKET, ROOT_CAUSE, ExplodingLLM())
    assert result["verdict"] == "NON_RESPONSIVE"
    assert "failing closed" in result["reason"]


def test_malformed_json_fails_closed_to_non_responsive():
    llm = ScriptedLLM([{"text": "not json at all"}])
    assert check_responsiveness(TICKET, ROOT_CAUSE, llm)["verdict"] == "NON_RESPONSIVE"


def test_invalid_verdict_enum_fails_closed_to_non_responsive():
    llm = ScriptedLLM([_resp("MOSTLY_FINE")])
    result = check_responsiveness(TICKET, ROOT_CAUSE, llm)
    assert result["verdict"] == "NON_RESPONSIVE"
    assert "failing closed" in result["reason"]


def test_verifier_style_verdict_is_not_accepted():
    """SUPPORTED is a valid verdict in the other module and must not leak
    into this one - the two vocabularies are separate on purpose."""
    llm = ScriptedLLM([_resp("SUPPORTED")])
    assert check_responsiveness(TICKET, ROOT_CAUSE, llm)["verdict"] == "NON_RESPONSIVE"


def test_empty_root_cause_is_non_responsive_without_calling_the_model():
    llm = ScriptedLLM([])  # would raise if called
    result = check_responsiveness(TICKET, "   ", llm)
    assert result["verdict"] == "NON_RESPONSIVE"
    assert llm.calls == []


def test_empty_ticket_is_non_responsive_without_calling_the_model():
    llm = ScriptedLLM([])
    assert check_responsiveness("", ROOT_CAUSE, llm)["verdict"] == "NON_RESPONSIVE"
    assert llm.calls == []


# --- the isolation property ------------------------------------------------

def test_prompt_contains_the_ticket_and_the_root_cause():
    llm = ScriptedLLM([_resp("NON_RESPONSIVE")])
    check_responsiveness(TICKET, ROOT_CAUSE, llm)
    prompt = llm.calls[0][0]["text"]
    assert TICKET in prompt
    assert ROOT_CAUSE in prompt


def test_check_cannot_see_evidence_other_claims_or_state():
    """check_responsiveness takes two strings, so there is no parameter
    through which evidence could arrive. This asserts the caller-side
    wrapper honours the same boundary: given a whole diagnosis object
    stuffed with evidence content, sibling claims and merchant state, the
    prompt carries the root cause text and none of the rest."""
    llm = ScriptedLLM([_resp("RESPONSIVE")])
    ticket = {
        "id": "t1",
        "body": TICKET,
        "true_root_cause": "LABEL_LEAK_SENTINEL",
        "correct_routing": "ROUTING_LEAK_SENTINEL",
    }
    diagnosis = {
        "root_cause": {"text": ROOT_CAUSE, "evidence": ["doc:settlements#c2"]},
        "claims": [
            {"text": "SIBLING_CLAIM_SENTINEL", "evidence": ["doc:settlements#c9"]},
            {"text": "ANOTHER_CLAIM_SENTINEL", "evidence": []},
        ],
        "_verification": {"root_cause_verdict": {"verdict": "VERIFIER_LEAK_SENTINEL"}},
        "risk_class": "money_movement",
    }

    check_diagnosis_responsiveness(ticket, diagnosis, llm)
    prompt = llm.calls[0][0]["text"]

    assert TICKET in prompt
    assert ROOT_CAUSE in prompt
    for leak in (
        "LABEL_LEAK_SENTINEL",        # the eval label
        "ROUTING_LEAK_SENTINEL",      # the eval label
        "SIBLING_CLAIM_SENTINEL",     # another claim
        "ANOTHER_CLAIM_SENTINEL",
        "VERIFIER_LEAK_SENTINEL",     # the other check's verdict
        "doc:settlements#c2",         # evidence ids
        "money_movement",
    ):
        assert leak not in prompt, f"{leak} leaked into the responsiveness prompt"


def test_prompt_tells_the_model_not_to_judge_truth():
    """If the model starts grading correctness, the two checks stop being
    orthogonal and the responsiveness signal is no longer separable."""
    llm = ScriptedLLM([_resp("RESPONSIVE")])
    check_responsiveness(TICKET, ROOT_CAUSE, llm)
    prompt = llm.calls[0][0]["text"].lower()
    assert "not checking whether the answer is true" in prompt
    assert "assume every factual statement" in prompt


def test_temperature_is_zero_and_schema_is_constrained():
    captured = {}

    class Recording:
        def generate_turn(self, turns, model=None, tools=None, response_json_schema=None, temperature=None):
            captured.update(schema=response_json_schema, temperature=temperature, tools=tools)
            return _resp("RESPONSIVE")

    check_responsiveness(TICKET, ROOT_CAUSE, Recording())
    assert captured["temperature"] == 0.0
    assert captured["tools"] is None
    assert captured["schema"]["properties"]["verdict"]["enum"] == [
        "RESPONSIVE",
        "PARTIALLY_RESPONSIVE",
        "NON_RESPONSIVE",
    ]
