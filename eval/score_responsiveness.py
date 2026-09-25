"""Scores the responsiveness check against the manual adjudication.

Answers exactly three questions, framed before the run:
  1. Of the WRONG root causes, how many does it flag?
  2. Of the CORRECT root causes, how many does it wrongly flag?
  3. Does it catch the four false auto-resolves the verifier cannot?

    python eval/score_responsiveness.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROWS = json.loads((ROOT / "eval" / "heldout_responsiveness.json").read_text())["rows"]
TARGETS = ["heldout_001", "heldout_002", "heldout_007", "heldout_008"]

FLAGGED = ("NON_RESPONSIVE", "PARTIALLY_RESPONSIVE")


def pct(n, d):
    return f"{n}/{d} = {n / d:.0%}" if d else f"{n}/0 = n/a"


def main():
    by_adj = {"CORRECT": [], "WRONG": [], "REFUSED": []}
    for r in ROWS:
        by_adj[r["adjudication"]].append(r)

    print("=" * 74)
    print(f"RESPONSIVENESS CHECK on {len(ROWS)} held-out tickets")
    print("=" * 74)

    print("\n-- Verdict distribution --")
    for v in ("RESPONSIVE", "PARTIALLY_RESPONSIVE", "NON_RESPONSIVE"):
        n = sum(1 for r in ROWS if r["responsiveness"] == v)
        print(f"  {v:<22} {pct(n, len(ROWS))}")

    print("\n-- Q1: of WRONG root causes, how many flagged? --")
    wrong = by_adj["WRONG"]
    for v in ("NON_RESPONSIVE", "PARTIALLY_RESPONSIVE"):
        hits = [r for r in wrong if r["responsiveness"] == v]
        print(f"  {v:<22} {pct(len(hits), len(wrong))}  {[r['id'] for r in hits]}")
    flagged = [r for r in wrong if r["responsiveness"] in FLAGGED]
    print(f"  {'FLAGGED (either)':<22} {pct(len(flagged), len(wrong))}")
    missed = [r for r in wrong if r["responsiveness"] == "RESPONSIVE"]
    print(f"  {'missed (RESPONSIVE)':<22} {pct(len(missed), len(wrong))}  {[r['id'] for r in missed]}")

    print("\n-- Q2: of CORRECT root causes, how many wrongly flagged? --")
    correct = by_adj["CORRECT"]
    for v in ("NON_RESPONSIVE", "PARTIALLY_RESPONSIVE"):
        hits = [r for r in correct if r["responsiveness"] == v]
        print(f"  {v:<22} {pct(len(hits), len(correct))}  {[r['id'] for r in hits]}")
    fp = [r for r in correct if r["responsiveness"] in FLAGGED]
    print(f"  {'FALSE FLAG (either)':<22} {pct(len(fp), len(correct))}")

    print("\n-- REFUSED (fail-closed, no causal claim; neither right nor wrong) --")
    for r in by_adj["REFUSED"]:
        print(f"  {r['id']}: {r['responsiveness']}")

    print("\n-- Q3: the four false auto-resolves the verifier cannot see --")
    caught = 0
    for tid in TARGETS:
        r = next(x for x in ROWS if x["id"] == tid)
        hit = r["responsiveness"] in FLAGGED
        caught += hit
        print(f"  {tid}: {r['responsiveness']:<22} {'CAUGHT' if hit else 'MISSED'}")
        print(f"      reason: {r['reason']}")
    print(f"  caught: {pct(caught, len(TARGETS))}")

    print("\n-- If NON_RESPONSIVE alone blocked AUTO_RESOLVE (not implemented) --")
    auto = [r for r in ROWS if r["gate_outcome"] == "AUTO_RESOLVE"]
    false_auto = [r for r in auto if r["adjudication"] == "WRONG"]
    for label, pred in (("NON_RESPONSIVE only", ("NON_RESPONSIVE",)), ("either flag", FLAGGED)):
        blocked_bad = [r for r in false_auto if r["responsiveness"] in pred]
        blocked_good = [r for r in auto if r["adjudication"] == "CORRECT" and r["responsiveness"] in pred]
        rate = (len(false_auto) - len(blocked_bad)) / len(ROWS)
        print(f"  {label:<22} false auto-resolve {len(false_auto)}/{len(ROWS)} -> "
              f"{len(false_auto) - len(blocked_bad)}/{len(ROWS)} ({rate:.0%}), "
              f"correct auto-resolves lost: {len(blocked_good)} {[r['id'] for r in blocked_good]}")


if __name__ == "__main__":
    main()
