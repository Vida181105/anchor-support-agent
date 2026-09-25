"""Measures src/responsiveness.py on the same 40 held-out tickets.

This does NOT re-run the agent. It replays the root cause the composer
already produced - `pre_verifier_root_cause` in
eval/heldout_results_verify_on.json, which is the composer's own text
before verify_diagnosis could rewrite it - against the ticket body, so
the responsiveness numbers are measured on exactly the diagnoses the
held-out evaluation scored. One LLM call per ticket, nothing else changes.

Correctness comes from eval/heldout_adjudication.json (manual), never
from either check. Grading a check with another model's output is the
circularity this evaluation has avoided throughout.

Nothing here reads or writes gate thresholds or gate rules.

    python eval/run_responsiveness.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.llm import LLMClient  # noqa: E402
from src.responsiveness import RESPONSIVENESS_MODEL, check_responsiveness  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "heldout_responsiveness.json"

FAIL_CLOSED_MARKER = "failing closed"

# The four false auto-resolves the check was commissioned to catch - the
# ones the verifier structurally cannot see, because it never reads the
# ticket. Named here before the run, not chosen after it.
TARGETS = ["heldout_001", "heldout_002", "heldout_007", "heldout_008"]


def main() -> None:
    tickets = {t["id"]: t for t in json.loads((ROOT / "eval" / "heldout.json").read_text())}
    on_rows = json.loads((ROOT / "eval" / "heldout_results_verify_on.json").read_text())["rows"]
    adj = json.loads((ROOT / "eval" / "heldout_adjudication.json").read_text())["judgements"]

    llm = LLMClient(max_retries=8, base_delay=4.0)
    rows = []
    for r in on_rows:
        tid = r["id"]
        root_cause = r.get("pre_verifier_root_cause") or ""
        verdict = check_responsiveness(tickets[tid]["body"], root_cause, llm)
        row = {
            "id": tid,
            "adjudication": adj[tid]["verdict"],
            "gate_outcome": r["gate_outcome"],
            "root_cause": root_cause,
            "responsiveness": verdict["verdict"],
            "reason": verdict["reason"],
            "fail_closed": FAIL_CLOSED_MARKER in verdict["reason"],
        }
        rows.append(row)
        print(f"{tid}: {row['adjudication']:<8} {row['responsiveness']:<22} {row['reason'][:60]}")
        OUT.write_text(
            json.dumps({"model": RESPONSIVENESS_MODEL, "complete": False, "rows": rows}, indent=2),
            encoding="utf-8",
        )

    OUT.write_text(
        json.dumps({"model": RESPONSIVENESS_MODEL, "complete": True, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    stuck = [r["id"] for r in rows if r["fail_closed"]]
    print(f"\nwrote {OUT} ({len(rows)}/{len(on_rows)})")
    if stuck:
        print(f"FAIL-CLOSED ARTIFACTS - must be re-run: {stuck}")


if __name__ == "__main__":
    main()
