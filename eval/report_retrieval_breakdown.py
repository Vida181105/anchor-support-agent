"""Breaks eval/retrieval_eval_results.json out by two independent axes -
ticket length band and Hinglish/code-mixed vs English - instead of the
single conflated "register" label used when the 25 tickets were first
selected. Run eval/run_retrieval_eval.py first to (re)generate the
results file this reads.

The Hinglish/English classification is a manual read, not a heuristic:
these are hand-written tickets from the training corpus and I know which
ones mix Hindi, so a wordlist guess would be strictly worse than just
recording the true label once.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "eval" / "retrieval_eval_results.json"
TICKETS_DIR = ROOT / "corpus" / "tickets"

HINGLISH_TICKET_IDS = {
    "ticket_094", "ticket_024", "ticket_091", "ticket_093", "ticket_099",
    "ticket_020", "ticket_035", "ticket_037", "ticket_041", "ticket_055",
    "ticket_069",
}


def word_count(ticket_id: str) -> int:
    body = json.loads((TICKETS_DIR / f"{ticket_id}.json").read_text())["body"]
    return len(body.split())


def length_band(n: int) -> str:
    if n < 15:
        return "short (<15w)"
    if n < 40:
        return "medium (15-40w)"
    return "long (40w+)"


def recall(rows, k):
    if not rows:
        return None
    return sum(r[f"hit@{k}"] for r in rows) / len(rows)


def report(title, rows):
    print(f"{title} (n={len(rows)})")
    for k in (1, 3, 5):
        r = recall(rows, k)
        print(f"  recall@{k}: {r:.0%}" if r is not None else f"  recall@{k}: n/a")


def main():
    results = json.loads(RESULTS_PATH.read_text())
    for r in results:
        n = word_count(r["ticket_id"])
        r["word_count"] = n
        r["length_band"] = length_band(n)
        r["language"] = "hinglish" if r["ticket_id"] in HINGLISH_TICKET_IDS else "english"

    bands = ("short (<15w)", "medium (15-40w)", "long (40w+)")

    print("=== Composition check (length band x language) ===")
    for band in bands:
        n_h = sum(1 for r in results if r["length_band"] == band and r["language"] == "hinglish")
        n_e = sum(1 for r in results if r["length_band"] == band and r["language"] == "english")
        print(f"  {band}: {n_h} hinglish, {n_e} english")
    print()

    print("=== Overall ===")
    report("all", results)
    print()

    print("=== By length band ===")
    for band in bands:
        report(band, [r for r in results if r["length_band"] == band])
    print()

    print("=== By language ===")
    report("hinglish/code-mixed", [r for r in results if r["language"] == "hinglish"])
    report("english", [r for r in results if r["language"] == "english"])
    print()

    print("=== recall@5 misses ===")
    for r in results:
        if not r["hit@5"]:
            print(f"  {r['ticket_id']} [{r['length_band']}, {r['language']}, {r['word_count']}w]")
            print(f"    wanted:    {r['correct']}")
            print(f"    retrieved: {r['retrieved_top5']}")


if __name__ == "__main__":
    main()
