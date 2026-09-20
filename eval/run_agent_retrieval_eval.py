"""State-informed retrieval evaluation: run the full agent loop
(src.agent.diagnose_ticket) over the same 25 hand-labeled tickets used for
the Phase 1 one-shot baseline (eval/retrieval_labels.json), and measure
recall@1/3/5 for the query the agent actually constructs and sends to
search_policy - not the raw ticket text.

A ticket can produce zero, one, or multiple search_policy calls. Recall is
computed per-ticket the same way as the baseline: a hit at k means at
least one hand-labeled correct chunk id appears in the top-k results of
AT LEAST ONE of the ticket's constructed queries (the most generous
reasonable reading - if the agent found the right chunk on any query it
issued, retrieval succeeded for that ticket).

Run: python eval/run_agent_retrieval_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.agent import diagnose_ticket  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.retrieval import build_index  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "eval" / "retrieval_labels.json"
TICKETS_DIR = ROOT / "corpus" / "tickets"
KS = (1, 3, 5)


def load_ticket(ticket_id: str) -> dict:
    return json.loads((TICKETS_DIR / f"{ticket_id}.json").read_text(encoding="utf-8"))


def hit_at_k(retrieved_id_lists: list[list[str]], correct_ids: set[str], k: int) -> bool:
    return any(set(ids[:k]) & correct_ids for ids in retrieved_id_lists)


def main() -> None:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    llm = LLMClient()
    index = build_index(llm)

    rows = []
    for label in labels:
        ticket = load_ticket(label["ticket_id"])
        correct = set(label["correct_chunks"])

        result = diagnose_ticket(ticket, llm, index)

        search_calls = [c for c in result["tool_call_log"] if c["tool"] == "search_policy"]
        constructed_queries = [c["args"]["query"] for c in search_calls]
        retrieved_id_lists = [[r["evidence_id"] for r in c["result"]] for c in search_calls]

        row = {
            "ticket_id": label["ticket_id"],
            "register": label["register"],
            "correct": sorted(correct),
            "identity_outcome": result["identity"]["outcome"],
            "num_search_calls": len(search_calls),
            "constructed_queries": constructed_queries,
            "retrieved_per_query": retrieved_id_lists,
            **{f"hit@{k}": hit_at_k(retrieved_id_lists, correct, k) if retrieved_id_lists else False for k in KS},
        }
        rows.append(row)
        print(f"done: {label['ticket_id']} ({len(search_calls)} search_policy calls)")

    def recall(rows_subset, k):
        if not rows_subset:
            return None
        return sum(r[f"hit@{k}"] for r in rows_subset) / len(rows_subset)

    print()
    print(f"{'ticket':<14} {'register':<16} hit@1 hit@3 hit@5  #queries  raw -> constructed")
    for r in rows:
        marks = "  ".join("Y" if r[f"hit@{k}"] else "." for k in KS)
        q_preview = r["constructed_queries"][0] if r["constructed_queries"] else "(no search_policy call)"
        print(f"{r['ticket_id']:<14} {r['register']:<16} {marks}    {r['num_search_calls']}   -> {q_preview}")

    print()
    print("=== Overall (n=%d) ===" % len(rows))
    for k in KS:
        r = recall(rows, k)
        print(f"recall@{k}: {r:.0%}" if r is not None else f"recall@{k}: n/a")

    print()
    hard = [r for r in rows if r["register"] in ("terse", "terse_hinglish")]
    easy = [r for r in rows if r["register"] not in ("terse", "terse_hinglish")]
    print(f"=== Terse/Hinglish subset (n={len(hard)}) ===")
    for k in KS:
        r = recall(hard, k)
        print(f"recall@{k}: {r:.0%}" if r is not None else "recall@{k}: n/a")
    print(f"=== Everything else (n={len(easy)}) ===")
    for k in KS:
        r = recall(easy, k)
        print(f"recall@{k}: {r:.0%}" if r is not None else "recall@{k}: n/a")

    zero_search = [r for r in rows if r["num_search_calls"] == 0]
    if zero_search:
        print()
        print(f"=== Tickets where the agent never called search_policy at all (n={len(zero_search)}) ===")
        for r in zero_search:
            print(f"  {r['ticket_id']} [{r['register']}] identity={r['identity_outcome']}")

    misses = [r for r in rows if not r["hit@5"]]
    if misses:
        print()
        print("=== Complete misses (correct chunk not in top 5 of ANY constructed query) ===")
        for r in misses:
            print(f"  {r['ticket_id']} [{r['register']}]: wanted {r['correct']}")
            for q, ids in zip(r["constructed_queries"], r["retrieved_per_query"]):
                print(f"    query: {q!r} -> {ids}")

    out_path = ROOT / "eval" / "agent_retrieval_eval_results.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote per-ticket results to {out_path}")


if __name__ == "__main__":
    main()
