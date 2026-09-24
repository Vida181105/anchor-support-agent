"""Scores the two held-out runs against the labels in eval/heldout.json
and the manual adjudication in eval/heldout_adjudication.json.

Nothing here reads the verifier's output to decide whether a root cause is
right - that judgement comes from the labels and the adjudication file.
Using the verifier to grade its own ablation would be circular, which is
the whole reason the headline metric is computed on PRE-GATE diagnoses
with correctness supplied externally.

Run: python eval/score_heldout.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADJ = json.loads((ROOT / "eval" / "heldout_adjudication.json").read_text())["judgements"]

# The labels predate the gate's fourth outcome. eval/README.md defines the
# correct action for an UNIDENTIFIABLE ticket as "a human sends a short
# reply asking for identification", which is exactly REQUEST_IDENTIFICATION
# - so it maps onto the draft_for_human label rather than contradicting it.
GATE_TO_LABEL = {
    "AUTO_RESOLVE": "auto_resolve",
    "DRAFT_FOR_HUMAN": "draft_for_human",
    "REQUEST_IDENTIFICATION": "draft_for_human",
    "ESCALATE": "escalate",
}


def load(mode):
    return json.loads((ROOT / "eval" / f"heldout_results_verify_{mode}.json").read_text())["rows"]


def pct(n, d):
    return f"{n}/{d} = {n / d:.0%}" if d else f"{n}/0 = n/a"


def score(rows, mode):
    print("=" * 74)
    print(f"VERIFIER {mode.upper()}  (n={len(rows)}, model gemini-3.5-flash-lite)")
    print("=" * 74)

    verdicts = {r["id"]: ADJ[r["id"]]["verdict"] for r in rows}
    correct = [r for r in rows if verdicts[r["id"]] == "CORRECT"]
    wrong = [r for r in rows if verdicts[r["id"]] == "WRONG"]
    refused = [r for r in rows if verdicts[r["id"]] == "REFUSED"]

    print("\n-- Root-cause accuracy (composer output; identical in both runs) --")
    print(f"  correct : {pct(len(correct), len(rows))}")
    print(f"  wrong   : {pct(len(wrong), len(rows))}")
    print(f"  refused : {pct(len(refused), len(rows))}  (no causal claim made)")
    print(f"  correct, excluding refusals: {pct(len(correct), len(correct) + len(wrong))}")

    print("\n-- Routing accuracy --")
    hits = [r for r in rows if GATE_TO_LABEL[r["gate_outcome"]] == r["label_routing"]]
    print(f"  overall : {pct(len(hits), len(rows))}")
    for lane in ("auto_resolve", "draft_for_human", "escalate"):
        want = [r for r in rows if r["label_routing"] == lane]
        got = [r for r in want if GATE_TO_LABEL[r["gate_outcome"]] == lane]
        print(f"    label={lane:<16} recall  {pct(len(got), len(want))}")
    for lane in ("auto_resolve", "draft_for_human", "escalate"):
        pred = [r for r in rows if GATE_TO_LABEL[r["gate_outcome"]] == lane]
        ok = [r for r in pred if r["label_routing"] == lane]
        print(f"    routed={lane:<16} precision {pct(len(ok), len(pred))}")

    print("\n-- Refusal accuracy --")
    for rtype, expected in (("EVIDENCE_GAP", "escalate"), ("UNIDENTIFIABLE", "draft_for_human")):
        sub = [r for r in rows if r["label_refusal_type"] == rtype]
        ok = [r for r in sub if GATE_TO_LABEL[r["gate_outcome"]] == expected]
        print(f"  {rtype:<16} routed correctly: {pct(len(ok), len(sub))}  (expected {expected})")
        if rtype == "UNIDENTIFIABLE":
            asked = [r for r in sub if r["gate_outcome"] == "REQUEST_IDENTIFICATION"]
            print(f"  {'':<16} of which via REQUEST_IDENTIFICATION: {pct(len(asked), len(sub))}")

    print("\n-- FALSE AUTO-RESOLVE (auto-resolved with a wrong root cause) --")
    auto = [r for r in rows if r["gate_outcome"] == "AUTO_RESOLVE"]
    false_auto = [r for r in auto if verdicts[r["id"]] == "WRONG"]
    print(f"  as a share of all tickets      : {pct(len(false_auto), len(rows))}")
    print(f"  as a share of auto-resolved    : {pct(len(false_auto), len(auto))}")
    if false_auto:
        print(f"  tickets: {[r['id'] for r in false_auto]}")
    return {"correct": correct, "wrong": wrong, "refused": refused, "false_auto": false_auto,
            "auto": auto, "verdicts": verdicts}


def headline(on_rows):
    """Pre-gate: does the verifier reject diagnoses in proportion to whether
    they are actually wrong? Correctness comes from the labels, not the
    verifier."""
    print("\n" + "=" * 74)
    print("HEADLINE - verifier behaviour on PRE-GATE diagnoses")
    print("=" * 74)
    rejected = lambda r: (r["verification"] or {}).get("root_cause_verdict") == "UNSUPPORTED"

    wrong = [r for r in on_rows if ADJ[r["id"]]["verdict"] == "WRONG"]
    correct = [r for r in on_rows if ADJ[r["id"]]["verdict"] == "CORRECT"]
    caught = [r for r in wrong if rejected(r)]
    false_rej = [r for r in correct if rejected(r)]

    print(f"  of WRONG root causes, verifier rejected   : {pct(len(caught), len(wrong))}")
    if caught:
        print(f"    caught: {[r['id'] for r in caught]}")
    print(f"  of CORRECT root causes, wrongly rejected  : {pct(len(false_rej), len(correct))}")
    if false_rej:
        print(f"    false rejections: {[r['id'] for r in false_rej]}")


def exceptions(on_rows, off_rows):
    print("\n" + "=" * 74)
    print("EXCEPTION LIST - every ticket with a wrong root cause")
    print("=" * 74)
    off = {r["id"]: r for r in off_rows}
    for r in on_rows:
        adj = ADJ[r["id"]]
        if adj["verdict"] != "WRONG":
            continue
        lbl = r["label_root_cause"]
        lbl = lbl if isinstance(lbl, str) else " + ".join(lbl)
        print(f"\n{r['id']}  [{r['label_refusal_type'] or 'normal'}]  merchant={r['merchant_id']}")
        print(f"  TRUE : {lbl}")
        print(f"  SAID : {(off[r['id']]['delivered_root_cause'] or '')[:200]}")
        print(f"  why wrong: {adj['note']}")
        print(f"  gate ON : {r['gate_outcome']} ({r['gate_rule']})")
        print(f"  gate OFF: {off[r['id']]['gate_outcome']} ({off[r['id']]['gate_rule']})")
        rc = (r["verification"] or {}).get("root_cause_verdict")
        print(f"  verifier root-cause verdict: {rc}")


def main():
    on_rows, off_rows = load("on"), load("off")
    s_on = score(on_rows, "on")
    print()
    s_off = score(off_rows, "off")
    headline(on_rows)

    print("\n" + "=" * 74)
    print("DELTA (ON vs OFF)")
    print("=" * 74)
    fa_on = len(s_on["false_auto"]) / len(on_rows)
    fa_off = len(s_off["false_auto"]) / len(off_rows)
    print(f"  false auto-resolve rate, all tickets : OFF {fa_off:.0%} -> ON {fa_on:.0%}")
    print(f"  auto-resolved count                  : OFF {len(s_off['auto'])} -> ON {len(s_on['auto'])}")

    exceptions(on_rows, off_rows)


if __name__ == "__main__":
    main()
