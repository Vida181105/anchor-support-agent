"""Units 3 & 4: run all 40 held-out tickets through the full pipeline and
the gate, with the verifier ON or OFF.

    python eval/run_heldout.py on
    python eval/run_heldout.py off

Thresholds were frozen 2026-09-23 10:58:05 IST (see eval/README.md) before
this script was ever run. Nothing here reads or writes them.

For the ON run this also records the PRE-VERIFIER diagnosis - the
composer's output before verify_diagnosis could reject or strip anything.
That costs nothing: the composer's turns are already in the disk cache
from the ON pass, so the verify=False call is pure cache hits. It is what
the headline metric needs, because a rejected diagnosis has its root_cause
text replaced by the rejection notice, so the ON result alone cannot say
what the composer originally concluded.

Resumable: every LLM call is disk-cached and results are written after
every ticket, so a 429/503 interruption costs only the remainder.
Fail-closed verifier artifacts (an API error surfacing as UNSUPPORTED) are
flagged per ticket so they can be re-run rather than recorded as verdicts.
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
HELDOUT = ROOT / "eval" / "heldout.json"

FAIL_CLOSED_MARKER = "verifier error, failing closed"


def fail_closed_artifacts(diagnosis: dict) -> list[str]:
    """Verdicts that are infrastructure noise, not judgements."""
    verification = diagnosis.get("_verification") or {}
    hits = []
    root = verification.get("root_cause_verdict") or {}
    if FAIL_CLOSED_MARKER in str(root.get("reason", "")):
        hits.append("root_cause")
    for v in verification.get("verdicts", []):
        if FAIL_CLOSED_MARKER in str(v.get("reason", "")):
            hits.append(str(v.get("claim", ""))[:60])
    return hits


def summarise_verification(diagnosis: dict) -> dict:
    verification = diagnosis.get("_verification") or {}
    if not verification:
        return {"ran": False}
    claim_verdicts = [v for v in verification.get("verdicts", []) if not v.get("is_root_cause")]
    supported = sum(1 for v in claim_verdicts if v.get("verdict") == "SUPPORTED")
    return {
        "ran": True,
        "root_cause_verdict": (verification.get("root_cause_verdict") or {}).get("verdict"),
        "root_cause_reason": (verification.get("root_cause_verdict") or {}).get("reason"),
        "n_claim_verdicts": len(claim_verdicts),
        "n_supported": supported,
        "n_stripped": sum(1 for v in claim_verdicts if v.get("verdict") == "UNSUPPORTED"),
        "supported_ratio": (supported / len(claim_verdicts)) if claim_verdicts else None,
    }


def main() -> None:
    mode = (sys.argv[1] if len(sys.argv) > 1 else "on").lower()
    if mode not in ("on", "off"):
        raise SystemExit("usage: run_heldout.py [on|off]")
    verify = mode == "on"
    out_path = ROOT / "eval" / f"heldout_results_verify_{mode}.json"

    # Batch patience: the default retry budget is sized for a blip, not a
    # sustained 503 outage (which killed a training run outright).
    llm = LLMClient(max_retries=8, base_delay=4.0)
    index = build_index(llm)
    tickets = json.loads(HELDOUT.read_text(encoding="utf-8"))

    rows = []
    for ticket in tickets:
        result = diagnose_ticket(ticket, llm, index, verify=verify)
        diagnosis = result["diagnosis"]
        decision = evaluate(diagnosis)

        row = {
            "id": ticket["id"],
            "merchant_id": ticket.get("merchant_id"),
            "label_root_cause": ticket.get("true_root_cause"),
            "label_routing": ticket.get("correct_routing"),
            "label_refusal_type": ticket.get("refusal_type"),
            "identity_status": diagnosis.get("identity_status"),
            "risk_class": diagnosis.get("risk_class"),
            "category": diagnosis.get("category"),
            "delivered_root_cause": (diagnosis.get("root_cause") or {}).get("text"),
            "n_claims": len(diagnosis.get("claims", [])),
            "verification": summarise_verification(diagnosis),
            "gate_outcome": decision.outcome,
            "gate_rule": decision.rule,
            "fail_closed": fail_closed_artifacts(diagnosis),
            "tools_called": [c["tool"] for c in result["tool_call_log"]],
        }

        if verify:
            # Free: the composer's turns are already cached from the call
            # above, so this is cache hits only. Needed because a rejected
            # diagnosis overwrites root_cause with the rejection notice.
            pre = diagnose_ticket(ticket, llm, index, verify=False)["diagnosis"]
            row["pre_verifier_root_cause"] = (pre.get("root_cause") or {}).get("text")
            row["pre_verifier_n_claims"] = len(pre.get("claims", []))

        rows.append(row)
        flag = f"  [FAIL-CLOSED: {row['fail_closed']}]" if row["fail_closed"] else ""
        print(f"{ticket['id']}: {decision.outcome:22} {decision.rule:46}{flag}")

        out_path.write_text(
            json.dumps(
                {"mode": mode, "model": DEFAULT_AGENT_MODEL, "complete": False, "rows": rows},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    out_path.write_text(
        json.dumps(
            {"mode": mode, "model": DEFAULT_AGENT_MODEL, "complete": True, "rows": rows},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path} ({len(rows)}/{len(tickets)})")
    stuck = [r["id"] for r in rows if r["fail_closed"]]
    if stuck:
        print(f"FAIL-CLOSED ARTIFACTS - must be re-run: {stuck}")


if __name__ == "__main__":
    main()
