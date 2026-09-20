"""State-first diagnosis loop.

Not retrieve-then-lookup. Phase 1's retrieval eval showed one-shot policy
search on raw ticket text gets recall@1 of 50% on short tickets, and that
ticket_035 specifically can never be fixed by better retrieval: "refund
status pending" correctly matches refund policy, but the real cause is an
undelivered webhook the merchant doesn't know about. The diagnosis only
exists in the join between the ticket, state, and policy - so the loop is:

    identify merchant -> gather state -> form a retrieval query INFORMED
    by state -> retrieve policy -> compose diagnosis

Identity resolution (src.identity.identify_merchant) runs first and is
NOT part of the model's tool-calling loop - it's precise, deterministic
business logic, not something to leave to an LLM's discretion. A MISMATCH
or UNIDENTIFIABLE outcome short-circuits before any state tool is ever
called: answering from the wrong merchant's state is the risk this whole
module exists to prevent, so the model is never even given the chance to
query one merchant's account while believing it's another's - every
state tool is bound to the identified merchant_id as a closure, not
passed as a model-supplied argument.
"""

from __future__ import annotations

import json
from typing import Any

from src.identity import identify_merchant
from src.llm import LLMClient
from src.retrieval import PolicyIndex
from src.schema import DIAGNOSIS_JSON_SCHEMA, validate_diagnosis
from src.state_tools import get_disputes, get_merchant_state, get_settlement_schedule, get_transactions

DEFAULT_AGENT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MAX_TOOL_CALLS = 6

_SHORT_CIRCUIT_DIAGNOSES = {
    "UNIDENTIFIABLE": {
        "category": "identity_unresolved",
        "risk_class": "informational",
        "claims": [],
        "root_cause": "The merchant could not be identified from the ticket - no valid merchant_id, "
        "reference, or business name resolves to an account.",
        "recommended_action": "draft_for_human",
    },
    "MISMATCH": {
        "category": "identity_mismatch",
        "risk_class": "money_movement",
        "claims": [],
        "root_cause": "The ticket's claimed merchant_id does not match the merchant identified via a "
        "cited reference or business name. Proceeding would risk exposing a different "
        "merchant's account data.",
        "recommended_action": "escalate",
    },
}

_FAIL_CLOSED_DIAGNOSIS = {
    "category": "insufficient_evidence",
    "risk_class": "informational",
    "claims": [],
    "root_cause": "The tool-call budget was exhausted before enough evidence was gathered to "
    "produce a grounded diagnosis.",
    "recommended_action": "escalate",
}

_UNGROUNDED_FAIL_CLOSED_DIAGNOSIS = {
    "category": "insufficient_evidence",
    "risk_class": "informational",
    "claims": [],
    "root_cause": "The model's diagnosis cited evidence that was never actually gathered, even "
    "after one corrective retry.",
    "recommended_action": "escalate",
}

_INSTRUCTIONS = """\
You are Anchor, a payments support diagnosis assistant. Diagnose the \
merchant support ticket below using ONLY the tools provided - never from \
general knowledge about payment platforms.

Ticket (merchant_id={merchant_id}):
\"\"\"
{body}
\"\"\"

You have tools to look up this merchant's account state (settlement \
schedule, transactions, disputes, or the whole record) and to search the \
policy knowledge base.

CRITICAL: when you call search_policy, the query must be built from FACTS \
YOU HAVE ALREADY GATHERED from the state tools (for example "T+3 \
settlement cycle, batch captured 2 business days ago"), never copied or \
lightly paraphrased from the ticket text above. The ticket's own words are \
often misleading about the real cause - a merchant describing "refund \
status pending" may actually be facing an undelivered webhook, which \
policy search on that phrase will never find. Gather state FIRST, then \
search policy with a query informed by what you found.

Call whatever tools you need, in any order. Stop calling tools once you \
have enough evidence to answer; you do not need to call every tool.\
"""

_FINAL_PROMPT = """\
Produce the final diagnosis now as the structured JSON object described. \
Every claim's "evidence" list must contain only evidence_id values that \
appear verbatim in a tool result above - never invent one, never cite the \
ticket text itself, never leave a claim with an empty evidence list.\
"""

_CORRECTIVE_SUFFIX = """\

Your previous attempt cited an evidence_id that never appeared in any tool \
result above. Only cite evidence_id values that are literally present in \
the tool results shown in this conversation.\
"""

_TOOL_DECLARATIONS = [
    {
        "name": "get_merchant_state",
        "description": "The merchant's entire account record: business profile, KYC, "
        "settlement schedule, hold, reserve, international payments status, disputes, transactions.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_settlement_schedule",
        "description": "The merchant's settlement cycle, last/next payout, and any hold or reserve "
        "impact on it.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_transactions",
        "description": "The merchant's transactions, optionally filtered.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "captured | failed | refunded"},
                "method": {"type": "string", "description": "card | upi | netbanking | wallet"},
                "failure_code": {"type": "string", "description": "e.g. RISK_BLOCK, DO_NOT_HONOR"},
                "start_date": {"type": "string", "description": "YYYY-MM-DD, inclusive"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD, inclusive"},
            },
        },
    },
    {
        "name": "get_disputes",
        "description": "All disputes/chargebacks on the merchant's account.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "search_policy",
        "description": "Semantic search over the policy knowledge base. Query must be built from "
        "gathered state facts, not copied from the raw ticket text.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


def _find_evidence_ids(obj: Any) -> list[str]:
    ids = []
    if isinstance(obj, dict):
        if "evidence_id" in obj and isinstance(obj["evidence_id"], str):
            ids.append(obj["evidence_id"])
        for v in obj.values():
            ids.extend(_find_evidence_ids(v))
    elif isinstance(obj, list):
        for v in obj:
            ids.extend(_find_evidence_ids(v))
    return ids


def _build_tool_executor(merchant_id: str, llm: LLMClient, policy_index: PolicyIndex, retrieval_log: list):
    def execute(name: str, args: dict) -> Any:
        if name == "get_merchant_state":
            return get_merchant_state(merchant_id)
        if name == "get_settlement_schedule":
            return get_settlement_schedule(merchant_id)
        if name == "get_transactions":
            filters = {k: v for k, v in args.items() if v}
            return get_transactions(merchant_id, filters=filters or None)
        if name == "get_disputes":
            return get_disputes(merchant_id)
        if name == "search_policy":
            query = args["query"]
            retrieval_log.append({"constructed_query": query})
            return policy_index.search(query, llm, k=5)
        raise ValueError(f"unknown tool: {name!r}")

    return execute


def diagnose_ticket(
    ticket: dict,
    llm: LLMClient,
    policy_index: PolicyIndex,
    model: str = DEFAULT_AGENT_MODEL,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> dict:
    """Run the full state-first diagnosis loop for one ticket.

    Returns {"diagnosis", "identity", "tool_call_log", "retrieval_log",
    "evidence_pool"}. `retrieval_log` pairs the raw ticket body with every
    query actually sent to search_policy, so the difference between the
    two is directly inspectable - not just asserted in a docstring.
    """
    identity = identify_merchant(ticket)
    outcome = identity["outcome"]

    retrieval_log: list[dict] = [{"raw_ticket": ticket.get("body", "")}]

    if outcome in _SHORT_CIRCUIT_DIAGNOSES:
        diagnosis = dict(_SHORT_CIRCUIT_DIAGNOSES[outcome])
        diagnosis["identity_status"] = outcome
        return {
            "diagnosis": diagnosis,
            "identity": identity,
            "tool_call_log": [],
            "retrieval_log": retrieval_log,
            "evidence_pool": [],
        }

    merchant_id = identity["identified_merchant_id"]
    executor = _build_tool_executor(merchant_id, llm, policy_index, retrieval_log)

    turns: list[dict] = [
        {"role": "user", "text": _INSTRUCTIONS.format(merchant_id=merchant_id, body=ticket.get("body", ""))}
    ]
    tool_call_log: list[dict] = []
    evidence_pool: list[str] = []
    tool_calls_made = 0
    failed_closed = False

    while True:
        result = llm.generate_turn(turns, model=model, tools=_TOOL_DECLARATIONS, temperature=0.0)

        if "function_call" not in result:
            break  # model believes it has enough evidence

        if tool_calls_made >= max_tool_calls:
            failed_closed = True
            break

        fc = result["function_call"]
        turns.append({"role": "model", "function_call": fc})

        tool_result = executor(fc["name"], fc.get("args", {}))
        tool_calls_made += 1
        tool_call_log.append({"tool": fc["name"], "args": fc.get("args", {}), "result": tool_result})
        evidence_pool.extend(_find_evidence_ids(tool_result))

        turns.append({"role": "function", "name": fc["name"], "response": {"result": tool_result}})

    if failed_closed:
        diagnosis = dict(_FAIL_CLOSED_DIAGNOSIS)
    else:
        diagnosis = _run_final_diagnosis_call(llm, model, turns, set(evidence_pool))

    diagnosis["identity_status"] = outcome
    return {
        "diagnosis": diagnosis,
        "identity": identity,
        "tool_call_log": tool_call_log,
        "retrieval_log": retrieval_log,
        "evidence_pool": evidence_pool,
    }


def _attempt_diagnosis(llm: LLMClient, model: str, turns: list[dict], known_evidence_ids: set[str]) -> dict | None:
    """One attempt at the final structured call. Returns the diagnosis if
    it parses and passes the grounding/enum checks, else None - covering
    a malformed-JSON response exactly the same as an ungrounded or
    invalid-enum one, since both call for the same corrective retry.
    """
    result = llm.generate_turn(turns, model=model, response_json_schema=DIAGNOSIS_JSON_SCHEMA, temperature=0.0)
    try:
        diagnosis = json.loads(result["text"])
        validate_diagnosis(diagnosis, known_evidence_ids)
        return diagnosis
    except (json.JSONDecodeError, ValueError):  # UngroundedClaimError is a ValueError
        return None


def _run_final_diagnosis_call(llm: LLMClient, model: str, turns: list[dict], known_evidence_ids: set[str]) -> dict:
    attempt_turns = turns + [{"role": "user", "text": _FINAL_PROMPT}]
    diagnosis = _attempt_diagnosis(llm, model, attempt_turns, known_evidence_ids)
    if diagnosis is not None:
        return diagnosis

    retry_turns = turns + [{"role": "user", "text": _FINAL_PROMPT + _CORRECTIVE_SUFFIX}]
    diagnosis = _attempt_diagnosis(llm, model, retry_turns, known_evidence_ids)
    if diagnosis is not None:
        return diagnosis

    return dict(_UNGROUNDED_FAIL_CLOSED_DIAGNOSIS)
