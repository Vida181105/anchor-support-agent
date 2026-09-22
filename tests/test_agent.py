"""Mocked control-flow tests for src/agent.py: identity short-circuiting,
tool binding safety, the tool-call cap failing closed, and the grounding
retry/fail-closed path. No network calls - the model's turns are scripted.
"""

import json

import pytest

from src.agent import diagnose_ticket
from src.schema import DIAGNOSIS_JSON_SCHEMA


class ScriptedLLM:
    """Returns generate_turn results from a fixed script, in order.
    Records every call's (turns, tools, response_json_schema) for
    assertions on what the agent actually sent.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_turn(self, turns, model=None, tools=None, response_json_schema=None, temperature=0.0):
        self.calls.append({"turns": turns, "tools": tools, "response_json_schema": response_json_schema})
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self.script.pop(0)

    def embed(self, text, task_type="RETRIEVAL_DOCUMENT", output_dimensionality=None):
        raise AssertionError("embed should not be called in these mocked tests")


class FakePolicyIndex:
    """search() returns fixed evidence, and records every query it was
    asked, so tests can assert search_policy's query construction.
    """

    def __init__(self, results=None):
        self.results = results or [
            {"evidence_id": "doc:settlement-cycles#c1", "content": "T+3 explanation", "source_type": "policy"}
        ]
        self.queries = []

    def search(self, query, llm, k=5):
        self.queries.append(query)
        return self.results


def _valid_final_response(evidence_id):
    diagnosis = {
        "category": "settlement_timing",
        "risk_class": "money_movement",
        "claims": [{"text": "Settlement is on schedule.", "evidence": [evidence_id]}],
        "root_cause": {"text": "T+3 cycle, within window.", "evidence": [evidence_id]},
        "recommended_action": "auto_resolve",
    }
    return {"text": json.dumps(diagnosis)}


# --- identity short-circuit: no LLM call at all ---------------------------

def test_mismatch_short_circuits_before_any_llm_call():
    llm = ScriptedLLM([])  # would raise if called
    policy_index = FakePolicyIndex()
    ticket = json.load(open("corpus/tickets/ticket_017.json"))

    result = diagnose_ticket(ticket, llm, policy_index, verify=False)

    assert result["identity"]["outcome"] == "MISMATCH"
    assert result["diagnosis"]["identity_status"] == "MISMATCH"
    assert result["diagnosis"]["recommended_action"] == "escalate"
    assert result["diagnosis"]["claims"] == []
    # risk_class is the diagnostic taxonomy, not an identity-risk smuggling
    # channel: a MISMATCH never gets far enough to diagnose a subject, so
    # it stays "informational" - escalation comes from identity_status
    # alone, which a gate can read independently.
    assert result["diagnosis"]["risk_class"] == "informational"
    assert result["tool_call_log"] == []
    assert llm.calls == []


def test_unidentifiable_short_circuits_before_any_llm_call():
    llm = ScriptedLLM([])
    policy_index = FakePolicyIndex()
    ticket = json.load(open("corpus/tickets/ticket_016.json"))

    result = diagnose_ticket(ticket, llm, policy_index, verify=False)

    assert result["identity"]["outcome"] == "UNIDENTIFIABLE"
    assert result["diagnosis"]["recommended_action"] == "draft_for_human"
    assert llm.calls == []


# --- tool binding: model cannot query a different merchant ----------------

def test_tools_are_bound_to_identified_merchant_ignoring_any_model_supplied_id():
    # get_merchant_state's declared schema has no merchant_id parameter at
    # all, but even if a stray key showed up in args, the executor must
    # ignore it - the closed-over identified merchant_id is authoritative.
    script = [
        {"function_call": {"name": "get_merchant_state", "args": {"merchant_id": "merchant_99"}}},
        {"text": "done"},
        _valid_final_response("state:merchant_1"),
    ]
    llm = ScriptedLLM(script)
    policy_index = FakePolicyIndex()
    ticket = {"merchant_id": "merchant_1", "body": "money not come"}

    result = diagnose_ticket(ticket, llm, policy_index, verify=False)

    fetched = result["tool_call_log"][0]["result"]
    assert fetched["content"]["merchant_id"] == "merchant_1"  # not merchant_99


# --- cap enforcement fails closed, never guesses --------------------------

def test_exhausting_tool_call_budget_fails_closed_without_a_final_diagnosis_call():
    always_call_tool = {"function_call": {"name": "get_disputes", "args": {}}}
    script = [always_call_tool] * 10  # far more than any reasonable cap
    llm = ScriptedLLM(script)
    policy_index = FakePolicyIndex()
    ticket = {"merchant_id": "merchant_1", "body": "money not come"}

    result = diagnose_ticket(ticket, llm, policy_index, max_tool_calls=3, verify=False)

    assert result["diagnosis"]["category"] == "insufficient_evidence"
    assert result["diagnosis"]["recommended_action"] == "escalate"
    assert result["diagnosis"]["claims"] == []
    assert len(result["tool_call_log"]) == 3  # never exceeded the cap
    # exactly cap+1 generate_turn calls: 3 executed + 1 that hit the cap
    # and was never executed. No response_json_schema call was ever made -
    # the model was never asked to guess from partial evidence.
    assert len(llm.calls) == 4
    assert all(c["response_json_schema"] is None for c in llm.calls)


# --- grounding validation: retry once, then fail closed -------------------

def test_ungrounded_claim_triggers_one_corrective_retry_then_succeeds():
    script = [
        {"function_call": {"name": "get_settlement_schedule", "args": {}}},
        {"text": "done"},
        _valid_final_response("state:merchant_1.made_up_evidence"),  # ungrounded
        _valid_final_response("state:merchant_1.settlement_schedule"),  # grounded on retry
    ]
    llm = ScriptedLLM(script)
    policy_index = FakePolicyIndex()
    ticket = {"merchant_id": "merchant_1", "body": "money not come"}

    result = diagnose_ticket(ticket, llm, policy_index, verify=False)

    assert result["diagnosis"]["claims"][0]["evidence"] == ["state:merchant_1.settlement_schedule"]
    # the corrective retry's prompt must reference the correction
    retry_call = llm.calls[-1]
    assert "previous attempt cited" in retry_call["turns"][-1]["text"]


def test_malformed_json_from_model_triggers_retry_then_succeeds():
    script = [
        {"function_call": {"name": "get_settlement_schedule", "args": {}}},
        {"text": "done"},
        {"text": "not valid json{{{"},  # malformed
        _valid_final_response("state:merchant_1.settlement_schedule"),
    ]
    llm = ScriptedLLM(script)
    policy_index = FakePolicyIndex()
    ticket = {"merchant_id": "merchant_1", "body": "money not come"}

    result = diagnose_ticket(ticket, llm, policy_index, verify=False)

    assert result["diagnosis"]["claims"][0]["evidence"] == ["state:merchant_1.settlement_schedule"]


def test_invalid_enum_value_from_model_triggers_retry_then_fails_closed_if_still_bad():
    bad_diagnosis = {
        "category": "x",
        "risk_class": "not_a_real_class",
        "claims": [],
        "root_cause": {"text": "x", "evidence": []},
        "recommended_action": "auto_resolve",
    }
    script = [
        {"text": "done"},  # no tool calls at all
        {"text": json.dumps(bad_diagnosis)},
        {"text": json.dumps(bad_diagnosis)},  # still bad on retry
    ]
    llm = ScriptedLLM(script)
    policy_index = FakePolicyIndex()
    ticket = {"merchant_id": "merchant_1", "body": "money not come"}

    result = diagnose_ticket(ticket, llm, policy_index, verify=False)

    assert result["diagnosis"]["category"] == "insufficient_evidence"
    assert result["diagnosis"]["recommended_action"] == "escalate"


def test_persistently_ungrounded_claim_fails_closed_after_one_retry():
    script = [
        {"text": "done"},
        _valid_final_response("state:merchant_1.made_up_evidence_1"),
        _valid_final_response("state:merchant_1.made_up_evidence_2"),
    ]
    llm = ScriptedLLM(script)
    policy_index = FakePolicyIndex()
    ticket = {"merchant_id": "merchant_1", "body": "money not come"}

    result = diagnose_ticket(ticket, llm, policy_index, verify=False)

    assert result["diagnosis"]["category"] == "insufficient_evidence"
    assert result["diagnosis"]["recommended_action"] == "escalate"
    assert len(llm.calls) == 3  # tool-loop stop + 2 final-diagnosis attempts


# --- retrieval logging: raw ticket vs constructed query -------------------

def test_retrieval_log_captures_raw_ticket_and_constructed_query_separately():
    script = [
        {"function_call": {"name": "get_settlement_schedule", "args": {}}},
        {
            "function_call": {
                "name": "search_policy",
                "args": {"query": "T+3 settlement cycle, batch captured 2 business days ago"},
            }
        },
        {"text": "done"},
        _valid_final_response("state:merchant_1.settlement_schedule"),
    ]
    llm = ScriptedLLM(script)
    policy_index = FakePolicyIndex()
    ticket = {"merchant_id": "merchant_1", "body": "money not come pls check urgent"}

    result = diagnose_ticket(ticket, llm, policy_index, verify=False)

    raw_entry = result["retrieval_log"][0]
    query_entry = result["retrieval_log"][1]
    assert raw_entry["raw_ticket"] == "money not come pls check urgent"
    assert query_entry["constructed_query"] == "T+3 settlement cycle, batch captured 2 business days ago"
    assert query_entry["constructed_query"] != raw_entry["raw_ticket"]
    assert policy_index.queries == ["T+3 settlement cycle, batch captured 2 business days ago"]


# --- evidence pool accumulates across multiple tool calls ------------------

def test_evidence_pool_accumulates_ids_from_every_tool_call():
    script = [
        {"function_call": {"name": "get_settlement_schedule", "args": {}}},
        {"function_call": {"name": "get_disputes", "args": {}}},
        {"text": "done"},
        _valid_final_response("state:merchant_2.settlement_schedule"),
    ]
    llm = ScriptedLLM(script)
    policy_index = FakePolicyIndex()
    ticket = {"merchant_id": "merchant_2", "body": "reserve question"}

    result = diagnose_ticket(ticket, llm, policy_index, verify=False)

    assert "state:merchant_2.settlement_schedule" in result["evidence_pool"]
    assert "state:merchant_2.disputes[0]" in result["evidence_pool"]
    assert "state:merchant_2.disputes[1]" in result["evidence_pool"]


# --- the ablation switch: verify=True/False/None -----------------------

def _verdict_response(verdict, reason="because"):
    return {"text": json.dumps({"verdict": verdict, "reason": reason})}


def test_verify_true_runs_the_content_verifier_and_can_strip_a_claim():
    script = [
        {"function_call": {"name": "get_settlement_schedule", "args": {}}},
        {"text": "done"},
        _valid_final_response("state:merchant_1.settlement_schedule"),
        _verdict_response("SUPPORTED"),  # root_cause verdict
        _verdict_response("UNSUPPORTED", "contradicts the fetched schedule"),  # claim verdict
    ]
    llm = ScriptedLLM(script)
    policy_index = FakePolicyIndex()
    ticket = {"merchant_id": "merchant_1", "body": "money not come"}

    result = diagnose_ticket(ticket, llm, policy_index, verify=True)

    assert result["verified"] is True
    assert result["diagnosis"]["claims"] == []  # stripped by the content verifier
    assert result["diagnosis"]["_verification"]["root_cause_verdict"]["verdict"] == "SUPPORTED"
    assert len(llm.calls) == 5  # 2 tool-loop + 1 final diagnosis + 2 verifier calls


def test_verify_false_skips_the_content_verifier_entirely():
    script = [
        {"function_call": {"name": "get_settlement_schedule", "args": {}}},
        {"text": "done"},
        _valid_final_response("state:merchant_1.settlement_schedule"),
    ]
    llm = ScriptedLLM(script)
    policy_index = FakePolicyIndex()
    ticket = {"merchant_id": "merchant_1", "body": "money not come"}

    result = diagnose_ticket(ticket, llm, policy_index, verify=False)

    assert result["verified"] is False
    assert len(result["diagnosis"]["claims"]) == 1  # untouched by any verifier
    assert "_verification" not in result["diagnosis"]
    assert len(llm.calls) == 3  # no extra verifier calls made at all


def test_verify_none_falls_back_to_config_default(monkeypatch):
    monkeypatch.setattr("src.agent.VERIFIER_ENABLED_DEFAULT", False)
    script = [
        {"function_call": {"name": "get_settlement_schedule", "args": {}}},
        {"text": "done"},
        _valid_final_response("state:merchant_1.settlement_schedule"),
    ]
    llm = ScriptedLLM(script)
    policy_index = FakePolicyIndex()
    ticket = {"merchant_id": "merchant_1", "body": "money not come"}

    result = diagnose_ticket(ticket, llm, policy_index)  # verify not passed at all

    assert result["verified"] is False
    assert len(llm.calls) == 3  # config default (patched False) honored, no verifier calls


def test_short_circuit_diagnosis_never_runs_verifier_even_with_verify_true():
    llm = ScriptedLLM([])  # would raise if called at all
    policy_index = FakePolicyIndex()
    ticket = json.load(open("corpus/tickets/ticket_017.json"))  # MISMATCH

    result = diagnose_ticket(ticket, llm, policy_index, verify=True)

    assert result["verified"] is False
    assert llm.calls == []


def test_fail_closed_diagnosis_never_runs_verifier_even_with_verify_true():
    always_call_tool = {"function_call": {"name": "get_disputes", "args": {}}}
    script = [always_call_tool] * 10
    llm = ScriptedLLM(script)
    policy_index = FakePolicyIndex()
    ticket = {"merchant_id": "merchant_1", "body": "money not come"}

    result = diagnose_ticket(ticket, llm, policy_index, max_tool_calls=3, verify=True)

    assert result["verified"] is False
    assert result["diagnosis"]["category"] == "insufficient_evidence"
    assert len(llm.calls) == 4  # exactly the tool-loop calls, nothing more
