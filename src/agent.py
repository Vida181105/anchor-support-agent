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

Once a diagnosis is grounded (every citation points to evidence actually
gathered - src.schema.validate_diagnosis), it optionally passes through
src.verifier.verify_diagnosis: a genuinely separate pass that checks
whether the cited evidence's CONTENT actually supports each claim's TEXT,
which validate_diagnosis explicitly does not check. This is the
`verify` parameter / src.config.VERIFIER_ENABLED_DEFAULT ablation switch -
see diagnose_ticket's docstring.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.config import RESPONSIVENESS_ENABLED_DEFAULT, VERIFIER_ENABLED_DEFAULT
from src.derived_facts import get_derived_facts
from src.identity import identify_merchant
from src.llm import LLMClient
from src.responsiveness import check_responsiveness
from src.retrieval import PolicyIndex
from src.schema import DIAGNOSIS_JSON_SCHEMA, validate_diagnosis
from src.state_tools import get_disputes, get_merchant_state, get_settlement_schedule, get_transactions
from src.verifier import verify_diagnosis

DEFAULT_AGENT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MAX_TOOL_CALLS = 6

_SHORT_CIRCUIT_DIAGNOSES = {
    # risk_class stays the diagnostic taxonomy (does the SUBJECT MATTER
    # concern money movement?) even here - it's "informational" for both
    # of these because neither one ever gets far enough to diagnose a
    # subject at all. identity_status is what actually carries the risk
    # signal for these two, and it's set independently below (not folded
    # into risk_class): a Phase 4 gate reads identity_status directly to
    # decide MISMATCH deserves its own high bar, rather than inferring it
    # from a risk_class value that was never really about identity in the
    # first place. An identity mismatch escalates because of what it is,
    # not by borrowing a category it isn't.
    "UNIDENTIFIABLE": {
        "category": "identity_unresolved",
        "risk_class": "informational",
        "claims": [],
        "root_cause": {
            "text": "The merchant could not be identified from the ticket - no valid "
            "merchant_id, reference, or business name resolves to an account.",
            "evidence": [],
        },
        "recommended_action": "draft_for_human",
    },
    "MISMATCH": {
        "category": "identity_mismatch",
        "risk_class": "informational",
        "claims": [],
        "root_cause": {
            "text": "The ticket's claimed merchant_id does not match the merchant identified "
            "via a cited reference or business name. Proceeding would risk exposing a "
            "different merchant's account data.",
            "evidence": [],
        },
        "recommended_action": "escalate",
    },
}

_FAIL_CLOSED_DIAGNOSIS = {
    "category": "insufficient_evidence",
    "risk_class": "informational",
    "claims": [],
    "root_cause": {
        "text": "The tool-call budget was exhausted before enough evidence was gathered to "
        "produce a grounded diagnosis.",
        "evidence": [],
    },
    "recommended_action": "escalate",
}

_UNGROUNDED_FAIL_CLOSED_DIAGNOSIS = {
    "category": "insufficient_evidence",
    "risk_class": "informational",
    "claims": [],
    "root_cause": {
        "text": "The model's diagnosis cited evidence that was never actually gathered, even "
        "after one corrective retry.",
        "evidence": [],
    },
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

CRITICAL: never do date arithmetic yourself. Do not count business days, \
do not work out how many days have passed, do not decide whether a \
deadline or window has been reached. Call get_derived_facts, which \
computes all of that in code against a fixed current date, and cite the \
derived:... evidence id it returns. Any statement about a due date, \
elapsed time, or whether something is overdue or still inside a window \
must rest on a derived fact - never on your own counting.

Call whatever tools you need, in any order. Stop calling tools once you \
have enough evidence to answer; you do not need to call every tool.\
"""

_FINAL_PROMPT = """\
Produce the final diagnosis now as the structured JSON object described.

Every claim must be ATOMIC - exactly one checkable fact, citing only the \
evidence for that one fact:

- One fact per claim. If a sentence contains "and", or states two numbers, \
or gives a fact plus what it means, split it into separate claims.
- No business or merchant names in claim text. The evidence for a \
settlement cycle establishes the cycle, not whose cycle it is, so a claim \
naming the merchant can never be fully supported. Names belong in the \
customer-facing message, which is written later.
- No dates, amounts, codes or figures that do not appear in the evidence \
you cite for that claim.
- No explanatory framing, reassurance, or next steps inside a claim. State \
the fact only.
- Anything time-relative - a due date, days elapsed, whether a window has \
passed or a payout is overdue - must cite a derived:... evidence id. Never \
assert a computed date or duration from your own reasoning.

root_cause is claim-shaped and held to the same standard: one checkable \
assertion with its own "evidence" list.

Every "evidence" list, on root_cause and on every claim, must contain only \
evidence_id values that appear verbatim in a tool result above - never \
invent one, never cite the ticket text itself, never leave one empty.\
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
        "name": "get_derived_facts",
        "description": "Date arithmetic computed in code against a fixed current date: the "
        "expected settlement date for the oldest unsettled batch (business days, weekends "
        "excluded), business days elapsed since it was captured, whether settlement is "
        "overdue, days since the last payout, and for each dispute how many days have passed "
        "since evidence was submitted and whether that is still inside the bank's review "
        "window. Use this for ANY time-relative statement instead of counting yourself.",
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


def _find_evidence_content(obj: Any) -> dict[str, Any]:
    """Same walk as _find_evidence_ids, but keeps each envelope's content
    too - what the Unit 1 verifier needs to check a claim against, as
    opposed to validate_diagnosis's grounding check, which only needs ids.
    """
    content_by_id: dict[str, Any] = {}
    if isinstance(obj, dict):
        if "evidence_id" in obj and isinstance(obj["evidence_id"], str):
            content_by_id[obj["evidence_id"]] = obj.get("content")
        for v in obj.values():
            content_by_id.update(_find_evidence_content(v))
    elif isinstance(obj, list):
        for v in obj:
            content_by_id.update(_find_evidence_content(v))
    return content_by_id


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
        if name == "get_derived_facts":
            return get_derived_facts(merchant_id)
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
    verify: bool | None = None,
    check_responsive: bool | None = None,
) -> dict:
    """Run the full state-first diagnosis loop for one ticket.

    `verify` is the ablation switch: whether the Unit 1 grounding verifier
    (src.verifier.verify_diagnosis) runs over the result before it's
    returned. Defaults to src.config.VERIFIER_ENABLED_DEFAULT when not
    passed explicitly, so a batch script can run the same held-out set
    twice - verify=True, then verify=False - as a config change, not a
    code change. Verification never runs over a short-circuit (MISMATCH/
    UNIDENTIFIABLE) or fail-closed diagnosis: both already have empty
    claims and no real evidence to check content against, so running the
    verifier there would just reject a refusal that was already correct.

    `check_responsive` is the second, independent ablation switch
    (src.config.RESPONSIVENESS_ENABLED_DEFAULT): whether
    src.responsiveness.check_responsiveness runs over the root cause. It
    answers a different question than the verifier - not "is this claim
    true" but "is this an answer to what was asked" - and is deliberately
    blinded to the evidence, as the verifier is blinded to the ticket.

    The responsiveness check is fed the root cause text as the COMPOSER
    wrote it, captured before verify_diagnosis can rewrite it. That is
    load-bearing: a rejected diagnosis has its root_cause replaced by a
    rejection notice, so checking the post-verifier text would be judging
    the verifier's output rather than the agent's, and the two checks
    would no longer be independent. Neither check sees the other's verdict.

    Nothing in the gate reads `_responsiveness` yet; it is recorded so it
    can be measured before it is allowed to route anything.

    Returns {"diagnosis", "identity", "tool_call_log", "retrieval_log",
    "evidence_pool", "evidence_content", "verified",
    "responsiveness_checked"}. `evidence_content` maps every gathered
    evidence_id to the content behind it, so a citation can be resolved
    back to the policy chunk or state field it points at. `retrieval_log` pairs the raw ticket
    body with every query actually sent to search_policy, so the
    difference between the two is directly inspectable - not just
    asserted in a docstring. `verified` is True only when the content
    verifier actually ran (and is therefore reflected in the diagnosis).
    """
    if verify is None:
        verify = VERIFIER_ENABLED_DEFAULT
    if check_responsive is None:
        check_responsive = RESPONSIVENESS_ENABLED_DEFAULT

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
            "evidence_content": {},
            "verified": False,
            "responsiveness_checked": False,
        }

    merchant_id = identity["identified_merchant_id"]
    executor = _build_tool_executor(merchant_id, llm, policy_index, retrieval_log)

    turns: list[dict] = [
        {"role": "user", "text": _INSTRUCTIONS.format(merchant_id=merchant_id, body=ticket.get("body", ""))}
    ]
    tool_call_log: list[dict] = []
    evidence_pool: list[str] = []
    evidence_content: dict[str, Any] = {}
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
        evidence_content.update(_find_evidence_content(tool_result))

        turns.append({"role": "function", "name": fc["name"], "response": {"result": tool_result}})

    verified = False
    responsiveness_checked = False
    if failed_closed:
        diagnosis = dict(_FAIL_CLOSED_DIAGNOSIS)
    else:
        diagnosis, grounded = _run_final_diagnosis_call(llm, model, turns, set(evidence_pool))
        composed_root_cause = (diagnosis.get("root_cause") or {}).get("text", "")

        # The two checks are blinded to each other by design, which also
        # means neither's input depends on the other's output - so they
        # overlap. Responsiveness is one call and verification is several,
        # so responsiveness finishes inside the verifier's window and
        # costs nothing in wall time. Starting it first means its request
        # is already open while the verifier's root-cause call runs.
        responsiveness_future = None
        pool = None
        if grounded and check_responsive:
            pool = ThreadPoolExecutor(max_workers=1)
            # Two strings in, one verdict out. The signature is the
            # isolation: evidence_content is in scope here and never
            # reaches it.
            responsiveness_future = pool.submit(
                check_responsiveness, ticket.get("body", ""), composed_root_cause, llm
            )
        try:
            if grounded and verify:
                diagnosis = verify_diagnosis(diagnosis, evidence_content, llm)
                verified = True
            if responsiveness_future is not None:
                diagnosis["_responsiveness"] = {
                    **responsiveness_future.result(),
                    "checked_root_cause": composed_root_cause,
                }
                responsiveness_checked = True
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

    diagnosis["identity_status"] = outcome
    return {
        "diagnosis": diagnosis,
        "identity": identity,
        "tool_call_log": tool_call_log,
        "retrieval_log": retrieval_log,
        "evidence_pool": evidence_pool,
        "evidence_content": evidence_content,
        "verified": verified,
        "responsiveness_checked": responsiveness_checked,
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


def _run_final_diagnosis_call(
    llm: LLMClient, model: str, turns: list[dict], known_evidence_ids: set[str]
) -> tuple[dict, bool]:
    """Returns (diagnosis, grounded). `grounded` is False only for the
    fixed _UNGROUNDED_FAIL_CLOSED_DIAGNOSIS sentinel, so the caller can
    tell "a real diagnosis was produced" from "we gave up" without string-
    matching on category - that distinction is what decides whether the
    Unit 1 content verifier is even worth running (it has no real claims
    to check.
    """
    attempt_turns = turns + [{"role": "user", "text": _FINAL_PROMPT}]
    diagnosis = _attempt_diagnosis(llm, model, attempt_turns, known_evidence_ids)
    if diagnosis is not None:
        return diagnosis, True

    retry_turns = turns + [{"role": "user", "text": _FINAL_PROMPT + _CORRECTIVE_SUFFIX}]
    diagnosis = _attempt_diagnosis(llm, model, retry_turns, known_evidence_ids)
    if diagnosis is not None:
        return diagnosis, True

    return dict(_UNGROUNDED_FAIL_CLOSED_DIAGNOSIS), False
