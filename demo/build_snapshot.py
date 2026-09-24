"""Pre-computes every seeded ticket run to demo/snapshot.json.

The demo must work on first click after a cold start with no API key
present, so nothing the page shows on arrival may require a live call.
This script is the only thing that talks to the model; the server reads
its output. Every LLM call here is disk-cached, so re-running it is free.

    python demo/build_snapshot.py

Also folds in the held-out metrics (frozen and post-hoc, kept separate)
and the adjudication verdicts, so the metrics panel and the "known
failure" labelling are read from the same measured files the evaluation
scripts read - not retyped into the UI where they could drift.
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
from src.schema import render_prose  # noqa: E402
from src.state_tools import _load_merchant  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "snapshot.json"

# The queue. Chosen to cover all four gate lanes and to include the three
# pinned scenarios. `pinned` marks the three; `known_failure` marks cases
# the system gets WRONG, which are labelled as such in the UI rather than
# quietly dropped.
QUEUE = [
    {"src": "corpus", "id": "ticket_035", "pinned": "state_informed_retrieval",
     "blurb": "The ticket says \"refund status pending\". The real cause is an "
              "undelivered webhook. Pure retrieval on the merchant's own words "
              "structurally cannot find this - compare the two queries below."},
    {"src": "corpus", "id": "ticket_017", "pinned": "identity_mismatch",
     "blurb": "The submitted merchant_id is a claim, not a fact. It contradicts the "
              "business named in the body, so the agent refuses before a single "
              "state tool is called - it never reads the claimed merchant's data."},
    {"src": "heldout", "id": "heldout_008", "pinned": "evidence_gap_failure",
     "known_failure": True,
     "blurb": "KNOWN FAILURE. The merchant asked what evidence is needed for an "
              "UNAUTHORIZED claim specifically. The corpus does not document that. "
              "The agent answered confidently from the adjacent generic evidence "
              "list and auto-resolved, instead of naming the gap and escalating."},

    {"src": "heldout", "id": "heldout_002", "known_failure": True,
     "blurb": "KNOWN FAILURE (caught). The root cause is wrong - the merchant asked how "
              "to change their settlement bank account and was told the settlement cycle "
              "is T+3. Every claim is correctly grounded; none of them answer the "
              "question. The verifier passed it, because the verifier never sees the "
              "question. The post-hoc responsiveness rule is what stopped it."},
    {"src": "heldout", "id": "heldout_005", "known_failure": True,
     "blurb": "KNOWN FAILURE (caught). A fabricated confident negative - it asserted the "
              "platform does not provide POS machines. The corpus is silent on POS "
              "entirely, so this is the EVIDENCE_GAP failure mode again. The verifier "
              "rejected the root cause against its own cited evidence, so it escalated."},
    {"src": "heldout", "id": "heldout_035",
     "blurb": "A claim was stripped as unsupported, so the remainder goes to a human."},
    {"src": "heldout", "id": "heldout_009",
     "blurb": "No id, no business name, nothing resolvable. The missing thing is "
              "identity, so the answer is to ask - not to escalate."},
    {"src": "heldout", "id": "heldout_017",
     "blurb": "A clean auto-resolve: identity corroborated, every claim supported."},
    {"src": "heldout", "id": "heldout_021",
     "blurb": "An active fraud hold. Correct diagnosis, but a human sends this one."},
    {"src": "heldout", "id": "heldout_034",
     "blurb": "Policy-only answer, discloses no account data, auto-resolves."},
    {"src": "heldout", "id": "heldout_024", "known_failure": True,
     "blurb": "KNOWN FAILURE (caught). Attributed the complaint to a wallet refund "
              "timeline; the real cause is an undelivered webhook, same class as "
              "ticket_035. Both checks flagged it, so it escalated anyway."},
]


def load_tickets() -> dict:
    tickets = {}
    for path in sorted((ROOT / "corpus" / "tickets").glob("*.json")):
        t = json.loads(path.read_text(encoding="utf-8"))
        tickets[t["id"]] = t
    for t in json.loads((ROOT / "eval" / "heldout.json").read_text(encoding="utf-8")):
        tickets[t["id"]] = t
    return tickets


def agent_visible(ticket: dict) -> dict:
    """Strip eval-only labels. The UI shows the ticket as the agent saw it,
    and shows the label separately as the answer key - never mixed in."""
    hidden = {"true_root_cause", "correct_routing", "refusal_type", "evidence_gap_fact",
              "true_merchant_id", "identity_note"}
    return {k: v for k, v in ticket.items() if k not in hidden and not k.startswith("_")}


def build_row(entry: dict, tickets: dict, llm, index, adjudication: dict) -> dict:
    ticket = tickets[entry["id"]]
    result = diagnose_ticket(ticket, llm, index, verify=True, check_responsive=True)
    diagnosis = result["diagnosis"]
    decision = evaluate(diagnosis)

    verification = diagnosis.get("_verification") or {}
    responsiveness = diagnosis.get("_responsiveness") or {}

    merchant_id = result["identity"].get("identified_merchant_id")
    merchant_name = None
    if merchant_id:
        try:
            merchant_name = (_load_merchant(merchant_id) or {}).get("business_name")
        except Exception:  # noqa: BLE001 - a demo snapshot never dies on a lookup
            merchant_name = None

    raw = next((e["raw_ticket"] for e in result["retrieval_log"] if "raw_ticket" in e), "")
    constructed = [e["constructed_query"] for e in result["retrieval_log"] if "constructed_query" in e]

    return {
        **{k: entry[k] for k in ("pinned", "known_failure", "blurb") if k in entry},
        "id": ticket["id"],
        "ticket": agent_visible(ticket),
        "labels": {
            "true_root_cause": ticket.get("true_root_cause"),
            "correct_routing": ticket.get("correct_routing"),
            "refusal_type": ticket.get("refusal_type"),
            "evidence_gap_fact": ticket.get("evidence_gap_fact"),
            "identity_note": ticket.get("identity_note"),
            "adjudication": (adjudication.get(ticket["id"]) or {}).get("verdict"),
            "adjudication_note": (adjudication.get(ticket["id"]) or {}).get("note"),
        },
        "identity": {**result["identity"], "business_name": merchant_name},
        "retrieval": {"raw_ticket": raw, "constructed_queries": constructed},
        "tool_calls": [
            {"tool": c["tool"], "args": c["args"], "result": c["result"]}
            for c in result["tool_call_log"]
        ],
        "diagnosis": {
            "category": diagnosis.get("category"),
            "risk_class": diagnosis.get("risk_class"),
            "root_cause": diagnosis.get("root_cause"),
            "claims": diagnosis.get("claims", []),
            "recommended_action": diagnosis.get("recommended_action"),
        },
        "rendered_prose": (
            render_prose(diagnosis, merchant_name) if diagnosis.get("root_cause") else ""
        ),
        # Identity evidence is folded in so a corroboration chip resolves in
        # the UI exactly like a claim's citation does.
        "evidence_content": {
            **{e["evidence_id"]: e.get("content")
               for e in result["identity"].get("evidence", []) if e.get("evidence_id")},
            **result["evidence_content"],
        },
        "verification": {
            "ran": bool(verification),
            "root_cause_verdict": verification.get("root_cause_verdict"),
            "verdicts": verification.get("verdicts", []),
        },
        "responsiveness": {
            "ran": bool(responsiveness),
            "verdict": responsiveness.get("verdict"),
            "reason": responsiveness.get("reason"),
            "checked_root_cause": responsiveness.get("checked_root_cause"),
        },
        "gate": decision.as_dict(),
    }


def metrics() -> dict:
    """Read straight from the evaluation artifacts. Frozen and post-hoc are
    separate objects here and separate panels in the UI - the post-hoc rule
    saw the held-out set, and merging the two would hide that."""
    adj = json.loads((ROOT / "eval" / "heldout_adjudication.json").read_text())["judgements"]
    frozen = json.loads((ROOT / "eval" / "heldout_results_verify_on.json").read_text())["rows"]
    off = json.loads((ROOT / "eval" / "heldout_results_verify_off.json").read_text())["rows"]
    post = json.loads((ROOT / "eval" / "heldout_results_posthoc.json").read_text())["rows"]
    resp = json.loads((ROOT / "eval" / "heldout_responsiveness.json").read_text())["rows"]

    to_label = {"AUTO_RESOLVE": "auto_resolve", "DRAFT_FOR_HUMAN": "draft_for_human",
                "REQUEST_IDENTIFICATION": "draft_for_human", "ESCALATE": "escalate"}

    def summarise(rows):
        n = len(rows)
        routing = sum(1 for r in rows if to_label[r["gate_outcome"]] == r["label_routing"])
        auto = [r for r in rows if r["gate_outcome"] == "AUTO_RESOLVE"]
        false_auto = [r for r in auto if adj[r["id"]]["verdict"] == "WRONG"]
        lanes = {}
        for lane in ("auto_resolve", "draft_for_human", "escalate"):
            want = [r for r in rows if r["label_routing"] == lane]
            lanes[lane] = [sum(1 for r in want if to_label[r["gate_outcome"]] == lane), len(want)]
        refusal = {}
        for rtype, expected in (("EVIDENCE_GAP", "escalate"), ("UNIDENTIFIABLE", "draft_for_human")):
            sub = [r for r in rows if r["label_refusal_type"] == rtype]
            refusal[rtype] = [sum(1 for r in sub if to_label[r["gate_outcome"]] == expected), len(sub)]
        return {"n": n, "routing": [routing, n], "auto_resolved": [len(auto), n],
                "false_auto": [len(false_auto), n],
                "false_auto_of_auto": [len(false_auto), len(auto)],
                "false_auto_ids": [r["id"] for r in false_auto],
                "lanes": lanes, "refusal": refusal}

    verdicts = [adj[r["id"]]["verdict"] for r in frozen]
    n_correct = verdicts.count("CORRECT")
    n_wrong = verdicts.count("WRONG")
    rejected = [r["id"] for r in frozen
                if (r["verification"] or {}).get("root_cause_verdict") == "UNSUPPORTED"]
    flagged = {r["id"]: r["responsiveness"] for r in resp}

    return {
        "n": len(frozen),
        "model": DEFAULT_AGENT_MODEL,
        "frozen_at": "2026-09-23 10:58:05 IST",
        "accuracy": {"correct": n_correct, "wrong": n_wrong,
                     "refused": verdicts.count("REFUSED"), "n": len(frozen),
                     "excluding_refusals": [n_correct, n_correct + n_wrong]},
        "frozen": summarise(frozen),
        "verifier_off": summarise(off),
        "posthoc": summarise(post),
        "checks": {
            "verifier_caught": [sum(1 for i in rejected if adj[i]["verdict"] == "WRONG"), n_wrong],
            "verifier_false_reject": [sum(1 for i in rejected if adj[i]["verdict"] == "CORRECT"), n_correct],
            "responsiveness_caught": [
                sum(1 for i, v in flagged.items() if adj[i]["verdict"] == "WRONG" and v != "RESPONSIVE"), n_wrong],
            "responsiveness_false_flag": [
                sum(1 for i, v in flagged.items() if adj[i]["verdict"] == "CORRECT" and v != "RESPONSIVE"), n_correct],
            "overlap_ids": sorted(set(rejected) & {i for i, v in flagged.items() if v != "RESPONSIVE"}),
        },
    }


def main() -> None:
    llm = LLMClient(max_retries=8, base_delay=4.0)
    index = build_index(llm)
    tickets = load_tickets()
    adjudication = json.loads((ROOT / "eval" / "heldout_adjudication.json").read_text())["judgements"]

    rows = []
    for entry in QUEUE:
        row = build_row(entry, tickets, llm, index, adjudication)
        rows.append(row)
        print(f"{row['id']}: {row['gate']['outcome']:22} {row['gate']['rule']}")

    OUT.write_text(json.dumps(
        {"model": DEFAULT_AGENT_MODEL, "queue": rows, "metrics": metrics()},
        indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT} ({len(rows)} tickets)")


if __name__ == "__main__":
    main()
