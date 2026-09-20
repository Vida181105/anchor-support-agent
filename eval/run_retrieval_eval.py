"""Retrieval-only evaluation: recall@1/3/5 of the policy index against 25
hand-labeled tickets from the training corpus (eval/retrieval_labels.json).

This measures retrieval alone, before any agent exists: for each ticket,
its raw body text is used as the search query, and a "hit" at k means at
least one hand-labeled correct chunk id appears in the top-k results.

Run: python eval/run_retrieval_eval.py
(Requires GEMINI_API_KEY on a cold cache; free after that.)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.llm import LLMClient  # noqa: E402
from src.retrieval import build_index  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "eval" / "retrieval_labels.json"
TICKETS_DIR = ROOT / "corpus" / "tickets"
KS = (1, 3, 5)


def load_ticket_body(ticket_id: str) -> str:
    data = json.loads((TICKETS_DIR / f"{ticket_id}.json").read_text(encoding="utf-8"))
    return data["body"]


def hit_at_k(retrieved_ids: list[str], correct_ids: set[str], k: int) -> bool:
    return bool(set(retrieved_ids[:k]) & correct_ids)


def main() -> None:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    llm = LLMClient()
    index = build_index(llm)

    rows = []
    for label in labels:
        body = load_ticket_body(label["ticket_id"])
        results = index.search(body, llm, k=max(KS))
        retrieved_ids = [r["evidence_id"] for r in results]
        correct = set(label["correct_chunks"])
        row = {
            "ticket_id": label["ticket_id"],
            "register": label["register"],
            "correct": sorted(correct),
            "retrieved_top5": retrieved_ids,
            **{f"hit@{k}": hit_at_k(retrieved_ids, correct, k) for k in KS},
        }
        rows.append(row)

    def recall(rows_subset, k):
        if not rows_subset:
            return None
        return sum(r[f"hit@{k}"] for r in rows_subset) / len(rows_subset)

    print(f"{'ticket':<14} {'register':<16} hit@1 hit@3 hit@5  correct -> top result")
    for r in rows:
        marks = "".join("Y" if r[f"hit@{k}"] else "." for k in KS)
        print(
            f"{r['ticket_id']:<14} {r['register']:<16} "
            f"{'  '.join(marks)}    {r['correct']} -> {r['retrieved_top5'][0]}"
        )

    print()
    print("=== Overall (n=%d) ===" % len(rows))
    for k in KS:
        print(f"recall@{k}: {recall(rows, k):.0%}")

    print()
    hard = [r for r in rows if r["register"] in ("terse", "terse_hinglish")]
    easy = [r for r in rows if r["register"] not in ("terse", "terse_hinglish")]
    print(f"=== Terse/Hinglish subset (n={len(hard)}) ===")
    for k in KS:
        print(f"recall@{k}: {recall(hard, k):.0%}")
    print(f"=== Everything else (n={len(easy)}) ===")
    for k in KS:
        print(f"recall@{k}: {recall(easy, k):.0%}")

    misses = [r for r in rows if not r["hit@5"]]
    if misses:
        print()
        print("=== Complete misses (correct chunk not even in top 5) ===")
        for r in misses:
            print(f"  {r['ticket_id']} [{r['register']}]: wanted {r['correct']}, got {r['retrieved_top5']}")

    out_path = ROOT / "eval" / "retrieval_eval_results.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote per-ticket results to {out_path}")


if __name__ == "__main__":
    main()
