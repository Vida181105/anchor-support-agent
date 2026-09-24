"""Unit 2: run the full pipeline over a stratified sample of TRAINING
tickets, to (a) measure the effect of replacing the gate's
money_movement proxy with the account-data-disclosure rule, and (b) set
the gate's tunable thresholds.

Training tickets only. Nothing here reads eval/heldout.json.

Resumable by construction: every LLM call goes through the disk cache, so
a run interrupted by the free tier's daily quota picks up where it left
off (see tests/test_cache_resumability.py). Tickets whose verification
came back as a 429 fail-closed artifact are flagged `rate_limited` and
excluded from threshold-setting rather than recorded as verdicts.

Run: python eval/run_training_sample.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.agent import DEFAULT_AGENT_MODEL, diagnose_ticket  # noqa: E402
from src.gate import evaluate  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.retrieval import build_index  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TICKETS = ROOT / "corpus" / "tickets"
OUT_PATH = ROOT / "eval" / "training_sample_results.json"

# Stratified by the two things that drive the rule under test: whether the
# answer is expected to rest on account state or only on published policy,
# plus the identity outcomes that short-circuit (free - no LLM calls).
SAMPLE = [
    # state-grounded causes: answers should cite state:/derived:
    "ticket_001",  # settlement mid-cycle
    "ticket_002",  # reserve after chargeback spike (CONFIRMED via business name)
    "ticket_004",  # partial hold
    "ticket_008",  # refund deducted from a later batch
    "ticket_009",  # dispute inside review window
    "ticket_012",  # KYC illegible resubmission
    "ticket_013",  # bank-side settlement delay
    "ticket_014",  # chargeback fee (CONFIRMED via dispute id)
    "ticket_020",  # mandate paused (terse Hinglish)
    "ticket_035",  # webhook undelivered - informational topic, account data answer
    "ticket_041",  # risk block after international approval (terse Hinglish)
    # policy-only causes: answers should cite doc: alone
    "ticket_051",  # card refund timeline (Hinglish)
    "ticket_053",  # webhook retry schedule
    "ticket_055",  # partnership KYC documents (Hinglish)
    "ticket_059",  # dispute response window
    "ticket_061",  # DO_NOT_HONOR meaning
    "ticket_064",  # settlement excludes weekends (Hinglish)
    "ticket_072",  # no automatic payment retry
    "ticket_105",  # chargeback fee is flat
    # identity short-circuits: no LLM calls at all
    "ticket_016",  # UNIDENTIFIABLE
    "ticket_017",  # MISMATCH
]

# Any verifier verdict carrying this prefix is an infrastructure artifact,
# not a judgement: the verifier fails closed to UNSUPPORTED on API errors.
# Originally this only looked for "429"; a 503 "high demand" error produces
# exactly the same artifact and would have been silently recorded as a real
# UNSUPPORTED verdict.
FAIL_CLOSED_MARKER = "verifier error, failing closed"


def old_money_movement_rule_would_block(diagnosis: dict) -> bool:
    """The rule this change replaced, reimplemented locally purely for the
    before/after comparison. It is deliberately NOT in src/gate.py - the
    old rule is gone from the system; this is a measuring stick.
    """
    return (
        diagnosis.get("identity_status") == "UNCORROBORATED"
        and diagnosis.get("risk_class") == "money_movement"
    )


def cites_account_data(diagnosis: dict) -> bool:
    parts = list(diagnosis.get("claims", []))
    rc = diagnosis.get("root_cause")
    if isinstance(rc, dict):
        parts.append(rc)
    return any(
        e.startswith(("state:", "derived:"))
        for p in parts
        for e in p.get("evidence", [])
        if isinstance(e, str)
    )


def was_rate_limited(diagnosis: dict) -> bool:
    verification = diagnosis.get("_verification") or {}
    for v in verification.get("verdicts", []):
        if FAIL_CLOSED_MARKER in str(v.get("reason", "")):
            return True
    return FAIL_CLOSED_MARKER in str(verification.get("root_cause_verdict", {}).get("reason", ""))


def main() -> None:
    # A batch run needs far more patience than an interactive call. The
    # default budget (5 retries, 1s base => ~31s) is sized for "the API
    # blipped"; a sustained 503 "high demand" outage outlasts it and kills
    # the batch. 8 retries at a 4s base backs off 4,8,16...512s ~= 17
    # minutes, which rides out an overload without hammering the API.
    # Set here rather than in LLMClient's defaults so tests stay fast.
    llm = LLMClient(max_retries=8, base_delay=4.0)
    index = build_index(llm)

    rows = []
    for ticket_id in SAMPLE:
        ticket = json.loads((TICKETS / f"{ticket_id}.json").read_text(encoding="utf-8"))
        result = diagnose_ticket(ticket, llm, index, verify=True)
        diagnosis = result["diagnosis"]
        decision = evaluate(diagnosis)

        verification = diagnosis.get("_verification") or {}
        claim_verdicts = [v for v in verification.get("verdicts", []) if not v.get("is_root_cause")]
        supported = sum(1 for v in claim_verdicts if v.get("verdict") == "SUPPORTED")

        row = {
            "ticket_id": ticket_id,
            "true_root_cause": ticket.get("true_root_cause"),
            "identity_status": diagnosis.get("identity_status"),
            "risk_class": diagnosis.get("risk_class"),
            "category": diagnosis.get("category"),
            "diagnosed_root_cause": diagnosis.get("root_cause", {}).get("text"),
            "n_claims": len(diagnosis.get("claims", [])),
            "n_claim_verdicts": len(claim_verdicts),
            "n_supported": supported,
            "supported_ratio": (supported / len(claim_verdicts)) if claim_verdicts else None,
            "root_cause_verdict": verification.get("root_cause_verdict", {}).get("verdict"),
            "stripped": sum(1 for v in claim_verdicts if v.get("verdict") == "UNSUPPORTED"),
            "cites_account_data": cites_account_data(diagnosis),
            "gate_outcome": decision.outcome,
            "gate_rule": decision.rule,
            "old_rule_would_block": old_money_movement_rule_would_block(diagnosis),
            "rate_limited": was_rate_limited(diagnosis),
            "tools_called": [c["tool"] for c in result["tool_call_log"]],
        }
        rows.append(row)
        flag = "  [FAIL-CLOSED ARTIFACT - excluded]" if row["rate_limited"] else ""
        print(
            f"{ticket_id}: {decision.outcome:22} {decision.rule:42} "
            f"ratio={row['supported_ratio']}{flag}"
        )

        # Written after every ticket, not just at the end: the LLM cache
        # makes recomputation free, but a crash mid-batch should never
        # lose rows already computed, and partial state stays inspectable.
        OUT_PATH.write_text(
            json.dumps(
                {"model": DEFAULT_AGENT_MODEL, "complete": False, "rows": rows},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    OUT_PATH.write_text(
        json.dumps(
            {"model": DEFAULT_AGENT_MODEL, "complete": True, "rows": rows}, indent=2, default=str
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_PATH} ({len(rows)}/{len(SAMPLE)} tickets)")


if __name__ == "__main__":
    main()
