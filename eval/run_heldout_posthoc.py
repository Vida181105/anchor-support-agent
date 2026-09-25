"""POST-HOC re-run of the 40 held-out tickets with BOTH checks on.

This is NOT the frozen-gate held-out result. The gate rule
`non_responsive_root_cause` was added after eval/heldout_results_verify_on.json
was produced and read, so this run has an advantage that the frozen run
did not: the rule was designed knowing which tickets it would face. It is
written to its own file and reported side by side, never merged.

Writes eval/heldout_results_posthoc.json.

    python eval/run_heldout_posthoc.py
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
from eval.run_heldout import fail_closed_artifacts, summarise_verification  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "heldout_results_posthoc.json"


def main() -> None:
    llm = LLMClient(max_retries=8, base_delay=4.0)
    index = build_index(llm)
    tickets = json.loads((ROOT / "eval" / "heldout.json").read_text(encoding="utf-8"))

    rows = []
    for ticket in tickets:
        result = diagnose_ticket(ticket, llm, index, verify=True, check_responsive=True)
        diagnosis = result["diagnosis"]
        decision = evaluate(diagnosis)
        resp = diagnosis.get("_responsiveness") or {}
        rows.append({
            "id": ticket["id"],
            "label_routing": ticket.get("correct_routing"),
            "label_refusal_type": ticket.get("refusal_type"),
            "delivered_root_cause": (diagnosis.get("root_cause") or {}).get("text"),
            "verification": summarise_verification(diagnosis),
            "responsiveness": resp.get("verdict"),
            "responsiveness_reason": resp.get("reason"),
            "gate_outcome": decision.outcome,
            "gate_rule": decision.rule,
            "fail_closed": fail_closed_artifacts(diagnosis),
            "responsiveness_fail_closed": "failing closed" in str(resp.get("reason", "")),
        })
        print(f"{ticket['id']}: {decision.outcome:22} {decision.rule}")
        OUT.write_text(json.dumps(
            {"mode": "posthoc_both_checks_on", "model": DEFAULT_AGENT_MODEL,
             "post_hoc_rules": ["non_responsive_root_cause"], "complete": False, "rows": rows},
            indent=2, default=str), encoding="utf-8")

    OUT.write_text(json.dumps(
        {"mode": "posthoc_both_checks_on", "model": DEFAULT_AGENT_MODEL,
         "post_hoc_rules": ["non_responsive_root_cause"], "complete": True, "rows": rows},
        indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT} ({len(rows)}/{len(tickets)})")
    stuck = [r["id"] for r in rows if r["fail_closed"] or r["responsiveness_fail_closed"]]
    if stuck:
        print(f"FAIL-CLOSED ARTIFACTS - must be re-run: {stuck}")


if __name__ == "__main__":
    main()
