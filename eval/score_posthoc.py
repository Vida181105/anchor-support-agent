"""Side-by-side: the FROZEN-GATE held-out result vs the POST-HOC re-run.

Kept deliberately separate from eval/score_heldout.py, which reports the
frozen numbers and must keep reporting them unchanged. The post-hoc rule
was added after those numbers were seen; merging the two would present a
rule that had already seen its test set as if it hadn't.

    python eval/score_posthoc.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADJ = json.loads((ROOT / "eval" / "heldout_adjudication.json").read_text())["judgements"]
FROZEN = {r["id"]: r for r in json.loads(
    (ROOT / "eval" / "heldout_results_verify_on.json").read_text())["rows"]}
POST = {r["id"]: r for r in json.loads(
    (ROOT / "eval" / "heldout_results_posthoc.json").read_text())["rows"]}

GATE_TO_LABEL = {
    "AUTO_RESOLVE": "auto_resolve",
    "DRAFT_FOR_HUMAN": "draft_for_human",
    "REQUEST_IDENTIFICATION": "draft_for_human",
    "ESCALATE": "escalate",
}


def pct(n, d):
    return f"{n}/{d} = {n / d:.0%}" if d else "n/a"


def stats(rows):
    ids = list(rows)
    hits = sum(1 for i in ids if GATE_TO_LABEL[rows[i]["gate_outcome"]] == rows[i]["label_routing"])
    auto = [i for i in ids if rows[i]["gate_outcome"] == "AUTO_RESOLVE"]
    false_auto = [i for i in auto if ADJ[i]["verdict"] == "WRONG"]
    return {"n": len(ids), "routing": hits, "auto": auto, "false_auto": false_auto}


def main():
    f, p = stats(FROZEN), stats(POST)

    print("=" * 78)
    print("FROZEN-GATE HELD-OUT RESULT vs POST-HOC RE-RUN")
    print("  frozen : gate rules frozen 2026-09-23 10:58:05 IST, before any held-out run")
    print("  posthoc: + non_responsive_root_cause, added AFTER the frozen result was read")
    print("=" * 78)

    rows = [
        ("routing accuracy", pct(f["routing"], f["n"]), pct(p["routing"], p["n"])),
        ("auto-resolved", pct(len(f["auto"]), f["n"]), pct(len(p["auto"]), p["n"])),
        ("FALSE auto-resolve (all tickets)", pct(len(f["false_auto"]), f["n"]), pct(len(p["false_auto"]), p["n"])),
        ("FALSE auto-resolve (of auto-resolved)",
         pct(len(f["false_auto"]), len(f["auto"])), pct(len(p["false_auto"]), len(p["auto"]))),
    ]
    print(f"\n  {'metric':<40} {'FROZEN':<16} {'POST-HOC':<16}")
    for name, a, b in rows:
        print(f"  {name:<40} {a:<16} {b:<16}")

    print("\n-- Per-lane recall --")
    for lane in ("auto_resolve", "draft_for_human", "escalate"):
        want = [i for i in FROZEN if FROZEN[i]["label_routing"] == lane]
        fo = sum(1 for i in want if GATE_TO_LABEL[FROZEN[i]["gate_outcome"]] == lane)
        po = sum(1 for i in want if GATE_TO_LABEL[POST[i]["gate_outcome"]] == lane)
        print(f"  {lane:<18} frozen {pct(fo, len(want)):<14} posthoc {pct(po, len(want))}")

    print("\n-- Every ticket whose OUTCOME changed --")
    changed = [i for i in FROZEN if FROZEN[i]["gate_outcome"] != POST[i]["gate_outcome"]]
    if not changed:
        print("  (none)")
    for i in sorted(changed):
        print(f"  {i}  [{ADJ[i]['verdict']}]  {FROZEN[i]['gate_outcome']} -> {POST[i]['gate_outcome']}")
        print(f"      rule: {FROZEN[i]['gate_rule']} -> {POST[i]['gate_rule']}")
        print(f"      responsiveness: {POST[i]['responsiveness']} - {POST[i]['responsiveness_reason']}")

    fired = [i for i in POST if POST[i]["gate_rule"] == "non_responsive_root_cause"]
    print(f"\n-- Where the post-hoc rule actually fired: {len(fired)} ticket(s) {sorted(fired)} --")
    absorbed = [i for i in POST
                if POST[i]["responsiveness"] == "NON_RESPONSIVE"
                and POST[i]["gate_rule"] != "non_responsive_root_cause"]
    print(f"   NON_RESPONSIVE but attributed to an earlier (frozen) rule: {len(absorbed)}")
    for i in sorted(absorbed):
        print(f"     {i}: {POST[i]['gate_outcome']:22} via {POST[i]['gate_rule']}")


if __name__ == "__main__":
    main()
